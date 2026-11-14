"""Scan & Bill — scan cans, save the bill, stock goes down at once; record customers who didn't buy."""
from html import escape
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import unquote

import pandas as pd
import streamlit as st

from src import billing, clock
from src.formatting import pack, rupees
from src.ui import data, nav
from src.ui.components import code_scanner
from src.ui.theme import chip, html, page_header

SS = st.session_state


def _title(text: str, sub: str = "") -> None:
    html(f'<div class="apx-step-t" style="font-size:18px">{escape(text)}</div>'
         + (f'<div class="apx-muted" style="margin:2px 0 8px">{sub}</div>' if sub else ""))


@st.cache_data(ttl=20, show_spinner=False)
def _products(tick: int) -> pd.DataFrame:      # tick changes after every bill -> fresh stock
    return billing.products()


def _catalogue() -> pd.DataFrame:
    return _products(SS.get("sb_tick", 0))


def _cart() -> List[Dict]:
    return SS.setdefault("sb_cart", [])


def _add(product_id: int, qty: int = 1) -> None:
    SS.setdefault("sb_undo", []).append((int(product_id), int(qty)))
    cart = _cart()
    for line in cart:
        if line["product_id"] == product_id:
            line["qty"] += qty
            return
    cart.append({"product_id": int(product_id), "qty": int(qty)})


def _changed_stock() -> None:
    """Stock and sales changed: refresh every screen's numbers."""
    SS["sb_tick"] = SS.get("sb_tick", 0) + 1
    data.clear_all()


def _notice(kind: str, text: str) -> None:
    """A short message shown over the camera."""
    SS["sb_notice_n"] = SS.get("sb_notice_n", 0) + 1
    SS["sb_notice"] = {"id": SS["sb_notice_n"], "text": text, "kind": kind}


def _cart_view() -> Dict:
    """The bill being made, as the scanner shows it (last scanned on top)."""
    cat = _catalogue().set_index("product_id")
    lines, total, count = [], 0.0, 0
    for line in _cart():
        if line["product_id"] not in cat.index:
            continue
        r = cat.loc[line["product_id"]]
        amount = line["qty"] * float(r["selling_price"])
        total += amount
        count += line["qty"]
        lines.append({"pid": int(line["product_id"]), "qty": int(line["qty"]), "label": str(r["label"]),
                      "hex": str(r["hex_code"]), "price": rupees(r["selling_price"]), "amount": rupees(amount),
                      "in_stock": int(r["in_stock"]), "short": int(r["in_stock"]) < line["qty"]})
    return {"count": count, "total": rupees(total), "lines": lines[::-1]}


def _done_view(bill: Optional[Dict]) -> Optional[Dict]:
    if not bill:
        return None
    link = billing.whatsapp_link(bill, "")
    return {"invoice": bill["invoice_no"], "time": clock.fmt_time(datetime.strptime(bill["created_at"], "%Y-%m-%d %H:%M:%S")),
            "total": rupees(bill["total"]), "payment": bill["payment_mode"],
            "lines": [{"qty": l["qty"], "label": l["label"], "amount": rupees(l["amount"])} for l in bill["lines"]],
            "wa_text": unquote(link.split("?text=", 1)[1])}


def _save(payment: str, allow_short: bool) -> None:
    cart = _cart()
    if not cart:
        return
    try:
        bill = billing.create_bill(cart, customer_type=SS.get("sb_ctype") or billing.CUSTOMER_TYPES[0],
                                   payment_mode=payment if payment in billing.PAYMENT_MODES else billing.PAYMENT_MODES[0],
                                   allow_short=allow_short)
    except billing.StockError as exc:
        _notice("warn", f"Not enough stock: {exc}")
        return
    SS["sb_cart"], SS["sb_undo"] = [], []
    SS["sb_done"] = bill
    SS["sb_pay"] = bill["payment_mode"]
    _changed_stock()


def _step(pid: int, by: int) -> None:
    cart = _cart()
    for i, line in enumerate(cart):
        if line["product_id"] == pid:
            line["qty"] += by
            if line["qty"] <= 0:
                cart.pop(i)
            return


def _undo() -> None:
    stack = SS.get("sb_undo") or []
    if stack:
        pid, qty = stack.pop()
        _step(pid, -qty)
        _notice("ok", "Removed the last can")


def _handle(val: Optional[Dict]) -> None:
    """One event from the scanner: a scan or a button."""
    if not val or not val.get("ts") or val["ts"] == SS.get("sb_last_scan"):
        return
    SS["sb_last_scan"] = val["ts"]
    act = val.get("action")
    if act == "save":
        _save(val.get("payment") or "Cash", allow_short=bool(val.get("force")))
    elif act == "undo":
        _undo()
    elif act in ("inc", "dec"):
        _step(int(val.get("pid", 0)), 1 if act == "inc" else -1)
    elif act == "next":
        SS["sb_done"] = None
    elif act == "phone":
        billing.set_phone(str(val.get("invoice", "")), str(val.get("phone", "")))
        return                                     # nothing on screen changes
    elif val.get("code"):
        hit = billing.lookup_code(val["code"])
        if hit:
            if SS.get("sb_done"):                   # scanning again = next customer
                SS["sb_done"] = None
            _add(int(hit["product_id"]))
            _notice("ok", f"+1 {hit['label']}")
        else:
            SS["sb_unknown"] = val["code"]
            _notice("warn", "New code — choose its product below")
    st.rerun()


# ----------------------------------------------------------------------------- the counter
def _counter() -> None:
    val = code_scanner(key="sb_scanner", cart=_cart_view(), done=_done_view(SS.get("sb_done")),
                       notice=SS.get("sb_notice"), payment=SS.get("sb_pay") or "Cash", payments=billing.PAYMENT_MODES)
    _handle(val)

    cat = _catalogue()
    opts = {int(r["product_id"]): f"{r['label']} · {r['brand']} · {rupees(r['selling_price'])} · {int(r['in_stock'])} in stock"
            for r in cat.to_dict("records")}
    unknown = SS.get("sb_unknown")
    if unknown:
        with st.container(border=True):
            html(f"<b>New code</b> <code>{escape(unknown)}</code> — which product is this? (asked only once)")
            pid = st.selectbox("Product", list(opts), format_func=opts.get, key="sb_link_pid",
                               label_visibility="collapsed", index=None, placeholder="Search a product…")
            c1, c2 = st.columns(2)
            if c1.button("Link & add", type="primary", disabled=pid is None, use_container_width=True, key="sb_link"):
                billing.link_code(unknown, int(pid))
                _add(int(pid))
                SS["sb_unknown"] = None
                _notice("ok", "Linked — this barcode now scans directly")
                st.rerun()
            if c2.button("Ignore", use_container_width=True, key="sb_ignore"):
                SS["sb_unknown"] = None
                st.rerun()

    with st.expander("No label? Add by name · customer type"):
        a, b, c = st.columns([5, 1.3, 1.4], vertical_alignment="bottom")
        with a:
            pid = st.selectbox("Product", list(opts), format_func=opts.get, key="sb_pick", index=None,
                               placeholder="Type a shade, brand or pack…", label_visibility="collapsed")
        with b:
            qty = st.number_input("Qty", 1, 200, 1, 1, key="sb_pick_qty", label_visibility="collapsed")
        with c:
            if st.button("Add", use_container_width=True, disabled=pid is None, key="sb_add"):
                SS["sb_done"] = None
                _add(int(pid), int(qty))
                _notice("ok", f"+{int(qty)} added")
                st.rerun()
        st.radio("Customer", billing.CUSTOMER_TYPES, horizontal=True, key="sb_ctype")


# ----------------------------------------------------------------------------- didn't buy
def _didnt_buy() -> None:
    with st.container(border=True):
        _title("Customer leaving without buying?", "One tap — shows where sales are lost")
        cat = _catalogue()
        opts = {int(r["product_id"]): f"{r['label']} · {r['brand']}" for r in cat.to_dict("records")}
        reasons = [(k, v) for k, v in billing.REASONS.items() if k != "other"]
        cols = st.columns(2) * ((len(reasons) + 1) // 2)
        for col, (key, text) in zip(cols, reasons):
            if col.button(text, key=f"sb_no_{key}", use_container_width=True):
                if key == "not_in_stock":
                    SS["sb_ask_shade"] = True
                else:
                    billing.record_no_sale(key)
                    SS["sb_no_flash"] = text
                    data.clear_all()
                    st.rerun()
        if SS.get("sb_ask_shade"):
            a, b = st.columns([4, 1.4], vertical_alignment="bottom")
            with a:
                want = st.selectbox("Which shade / pack did they want?", list(opts), format_func=opts.get, index=None,
                                    key="sb_no_pid", placeholder="Search (optional)")
            with b:
                if st.button("Save", type="primary", use_container_width=True, key="sb_no_save"):
                    billing.record_no_sale("not_in_stock", product_id=want)
                    SS["sb_ask_shade"] = False
                    SS["sb_no_flash"] = billing.REASONS["not_in_stock"] + (f" — {opts[want]}" if want else "")
                    data.clear_all()
                    st.rerun()
        if SS.get("sb_no_flash"):
            st.caption(f"Recorded: {SS['sb_no_flash']}")


# ----------------------------------------------------------------------------- today
def _today() -> None:
    bills = billing.bills_on()
    paid = bills[bills["status"] == "paid"] if not bills.empty else bills
    nos = billing.no_sales_on()
    sales = float(paid["total"].sum()) if not paid.empty else 0.0
    top = nos["reason"].value_counts().idxmax() if not nos.empty else None
    with st.container(border=True):
        _title("Today")
        c1, c2, c3 = st.columns(3)
        c1.metric("Bills", len(paid))
        c2.metric("Sales", rupees(sales))
        c3.metric("Didn't buy", len(nos), help=f"Most often: {billing.REASONS.get(top, top)}" if top else None)
        if bills.empty:
            st.caption("No bills yet today.")
            return
        with st.expander(f"Today's bills ({len(bills)})"):
            for r in bills.head(15).to_dict("records"):
                void = r["status"] == "void"
                c1, c2 = st.columns([5, 1.3], vertical_alignment="center")
                with c1:
                    html(f'<div class="apx-row" style="padding:3px 0">{chip("neutral" if void else "healthy", "Void" if void else "Paid")}'
                         f'<span class="apx-row-main"><b>{escape(r["invoice_no"][-3:])}</b> · {escape(r["created_at"][11:16])} · '
                         f'{r["items"]} can{"s" if r["items"] != 1 else ""} · {escape(r["payment_mode"])}</span>'
                         f'<span class="apx-row-right">{rupees(r["total"])}</span></div>')
                if not void and c2.button("Void", key=f"sb_void_{r['invoice_no']}", use_container_width=True,
                                          help="Cancel the bill and put the cans back into stock"):
                    billing.void_bill(r["invoice_no"])
                    if (SS.get("sb_done") or {}).get("invoice_no") == r["invoice_no"]:
                        SS["sb_done"] = None
                    _changed_stock()
                    st.rerun()


# ----------------------------------------------------------------------------- labels
def _labels() -> None:
    with st.expander("Print QR labels for cans and shelves"):
        cat = _catalogue()
        mode = st.radio("Which products", ["Fast movers (top 21)", "Choose…", "All products"], horizontal=True, key="sb_lbl_mode")
        if mode == "Choose…":
            opts = {int(r["product_id"]): f"{r['label']} · {r['brand']}" for r in cat.to_dict("records")}
            ids = st.multiselect("Products", list(opts), format_func=opts.get, key="sb_lbl_ids")
        elif mode == "All products":
            ids = [int(p) for p in cat["product_id"]]
        else:
            fm = data.cam_fast_movers("sample", clock.today_str())["top"]
            ids = [int(p) for p in fm["product_id"]][:21] or [int(p) for p in cat["product_id"]][:21]
        copies = st.number_input("Copies of each", 1, 50, 1, key="sb_lbl_copies")
        if ids:
            st.caption(f"{len(ids) * copies} labels · {-(-len(ids) * copies // 21)} A4 page(s) · 3 × 7 labels per page")
            pdf = billing.label_sheet(ids, copies=int(copies), fmt="PDF")
            st.download_button("Download labels (PDF)", pdf, file_name="apex_can_labels.pdf", mime="application/pdf",
                               use_container_width=True)


def render() -> None:
    page_header("Scan & Bill", "Scan the cans, save — stock goes down at once and the bill goes on WhatsApp.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        _counter()
    with right:
        _didnt_buy()
        _today()
        _labels()
        nav.link("camera", "Who bought — Store Camera →")
