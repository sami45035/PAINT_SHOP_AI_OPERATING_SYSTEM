"""Scan & Bill — scan cans, save the bill, stock goes down at once; record customers who didn't buy."""
from html import escape
from typing import Dict, List

import pandas as pd
import streamlit as st

from src import billing, clock
from src.formatting import pack, rupees
from src.ui import data, nav
from src.ui.components import code_scanner
from src.ui.theme import banner, chip, html, kpi_card, page_header

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


# ----------------------------------------------------------------------------- scanning
def _scanner() -> None:
    with st.container(border=True):
        _title("1 · Scan the cans", "QR labels from this app, or the barcode printed on the can (linked once)")
        val = code_scanner(key="sb_scanner")
        if val and val.get("ts") and val["ts"] != SS.get("sb_last_scan"):
            SS["sb_last_scan"] = val["ts"]
            hit = billing.lookup_code(val["code"])
            if hit:
                _add(int(hit["product_id"]))
                SS["sb_flash"] = ("healthy", f"Added {hit['label']} · {hit['brand']}")
            else:
                SS["sb_unknown"] = val["code"]
                SS["sb_flash"] = ("soon", f"Code {val['code']} isn't linked to a product yet")
        flash = SS.get("sb_flash")
        if flash:
            html(f'<div style="margin:6px 0">{chip(flash[0], "Scanned")} <span>{escape(flash[1])}</span></div>')

        unknown = SS.get("sb_unknown")
        cat = _catalogue()
        opts = {int(r["product_id"]): f"{r['label']} · {r['brand']} · {rupees(r['selling_price'])} · {int(r['in_stock'])} in stock"
                for r in cat.to_dict("records")}
        if unknown:
            with st.container(border=True):
                html(f"<b>New code:</b> <code>{escape(unknown)}</code> — which product is it? (asked only once)")
                pid = st.selectbox("Product", list(opts), format_func=opts.get, key="sb_link_pid",
                                   label_visibility="collapsed", index=None, placeholder="Search a product…")
                c1, c2 = st.columns(2)
                if c1.button("Link & add", type="primary", disabled=pid is None, use_container_width=True, key="sb_link"):
                    billing.link_code(unknown, int(pid))
                    _add(int(pid))
                    SS["sb_unknown"] = None
                    SS["sb_flash"] = ("healthy", f"Linked {unknown} and added it")
                    st.rerun()
                if c2.button("Ignore", use_container_width=True, key="sb_ignore"):
                    SS["sb_unknown"] = None
                    st.rerun()

        html('<div class="apx-label" style="margin-top:10px">No label? Add by name</div>')
        a, b, c = st.columns([5, 1.2, 1.4], vertical_alignment="bottom")
        with a:
            pid = st.selectbox("Product", list(opts), format_func=opts.get, key="sb_pick", index=None,
                               placeholder="Type a shade, brand or pack…", label_visibility="collapsed")
        with b:
            qty = st.number_input("Qty", 1, 200, 1, 1, key="sb_pick_qty", label_visibility="collapsed")
        with c:
            if st.button("Add", use_container_width=True, disabled=pid is None, key="sb_add"):
                _add(int(pid), int(qty))
                st.rerun()


# ----------------------------------------------------------------------------- the bill
def _bill() -> None:
    cart = _cart()
    cat = _catalogue().set_index("product_id")
    with st.container(border=True):
        _title("2 · The bill")
        if not cart:
            st.caption("Scan a can or add one by name — it appears here.")
        total = 0.0
        short = []
        for i, line in enumerate(list(cart)):
            if line["product_id"] not in cat.index:
                continue
            r = cat.loc[line["product_id"]]
            amount = line["qty"] * float(r["selling_price"])
            total += amount
            left = int(r["in_stock"]) - line["qty"]
            if left < 0:
                short.append(r["label"])
            c1, c2, c3, c4 = st.columns([5, 1, 1, 0.8], vertical_alignment="center")
            with c1:
                stock_note = (f'<span style="color:#B42318">only {int(r["in_stock"])} in stock</span>' if left < 0
                              else f"{left} left after this")
                html(f'<div class="apx-row" style="padding:2px 0"><span class="apx-sw" style="background:{escape(r["hex_code"])}"></span>'
                     f'<span class="apx-row-main"><b>{line["qty"]} × {escape(r["label"])}</b> · {escape(r["brand"])}'
                     f'<span class="apx-row-sub">{rupees(r["selling_price"])} each · {stock_note}</span></span>'
                     f'<span class="apx-row-right">{rupees(amount)}</span></div>')
            if c2.button("−", key=f"sb_minus_{i}", use_container_width=True):
                line["qty"] -= 1
                if line["qty"] <= 0:
                    cart.pop(i)
                st.rerun()
            if c3.button("+", key=f"sb_plus_{i}", use_container_width=True):
                line["qty"] += 1
                st.rerun()
            if c4.button("✕", key=f"sb_del_{i}", use_container_width=True):
                cart.pop(i)
                st.rerun()
        if cart:
            html(f'<div style="display:flex;justify-content:space-between;align-items:baseline;border-top:1px solid #E2E5EA;'
                 f'margin-top:8px;padding-top:10px"><span class="apx-label">Total · {sum(l["qty"] for l in cart)} cans</span>'
                 f'<span class="apx-kpi" style="font-size:28px">{rupees(total)}</span></div>')
            a, b = st.columns(2)
            with a:
                ctype = st.radio("Customer", billing.CUSTOMER_TYPES, horizontal=True, key="sb_ctype")
            with b:
                pay = st.radio("Payment", billing.PAYMENT_MODES, horizontal=True, key="sb_pay")
            phone = st.text_input("Customer's mobile (optional — to send the bill on WhatsApp)", key="sb_phone",
                                  placeholder="10-digit mobile")
            if short:
                banner("soon", "Stock", f"The stock book shows fewer cans than this bill for: {escape(', '.join(short))}. "
                                        "Recount the shelf, or save anyway and the stock goes to zero.")
            c1, c2 = st.columns([2, 1])
            if c1.button(f"Save bill · {rupees(total)}", type="primary", use_container_width=True, key="sb_save"):
                try:
                    bill = billing.create_bill(cart, customer_type=ctype, phone=phone, payment_mode=pay,
                                               allow_short=bool(short))
                except billing.StockError as exc:
                    st.error(f"Not enough stock: {exc}")
                    return
                SS["sb_cart"] = []
                SS["sb_done"] = bill
                SS["sb_flash"] = None
                _changed_stock()
                st.rerun()
            if c2.button("Clear", use_container_width=True, key="sb_clear"):
                SS["sb_cart"] = []
                st.rerun()

    done = SS.get("sb_done")
    if done:
        with st.container(border=True):
            html(f'{chip("healthy", "Saved")} <b>Bill {escape(done["invoice_no"])}</b> · {rupees(done["total"])} · '
                 f'{escape(done["payment_mode"])} · {escape(done["created_at"][11:16])}')
            rows = "".join(f'<div class="apx-row" style="padding:3px 0"><span class="apx-row-main">{l["qty"]} × {escape(l["label"])}'
                           f'<span class="apx-row-sub">stock now {l["stock_after"]}</span></span>'
                           f'<span class="apx-row-right">{rupees(l["amount"])}</span></div>' for l in done["lines"])
            html(rows + '<div class="apx-muted" style="margin-top:6px">Stock, sales, the forecast and alerts are updated. '
                        "The camera links this bill to the customer it saw at the counter.</div>")
            a, b = st.columns(2)
            a.link_button("Send on WhatsApp", billing.whatsapp_link(done, done.get("phone", "")), use_container_width=True)
            if b.button("New bill", use_container_width=True, key="sb_new"):
                SS["sb_done"] = None
                st.rerun()


# ----------------------------------------------------------------------------- didn't buy
def _didnt_buy() -> None:
    with st.container(border=True):
        _title("Customer leaving without buying?", "One tap — the reasons show you where sales are lost")
        cat = _catalogue()
        opts = {int(r["product_id"]): f"{r['label']} · {r['brand']}" for r in cat.to_dict("records")}
        cols = st.columns(len(billing.REASONS) - 1)
        for col, (key, text) in zip(cols, [(k, v) for k, v in billing.REASONS.items() if k != "other"]):
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
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Bills today", str(len(paid)), "made with Scan & Bill")
    with c2:
        kpi_card("Sales today", rupees(float(paid["total"].sum()) if not paid.empty else 0), "")
    with c3:
        top = nos["reason"].value_counts().idxmax() if not nos.empty else None
        kpi_card("Didn't buy", str(len(nos)), f"most often: {billing.REASONS.get(top, top).lower()}" if top else "none recorded")
    with st.container(border=True):
        _title("Today's bills")
        if paid.empty and (bills.empty or (bills["status"] == "void").sum() == 0):
            st.caption("No bills yet today.")
        for r in bills.head(12).to_dict("records"):
            void = r["status"] == "void"
            c1, c2 = st.columns([5, 1], vertical_alignment="center")
            with c1:
                html(f'<div class="apx-row" style="padding:3px 0">{chip("neutral" if void else "healthy", "Void" if void else "Paid")}'
                     f'<span class="apx-row-main"><b>{escape(r["invoice_no"])}</b> · {escape(r["created_at"][11:16])} · '
                     f'{r["items"]} can{"s" if r["items"] != 1 else ""} · {escape(r["customer_type"])}</span>'
                     f'<span class="apx-row-right">{rupees(r["total"])}</span></div>')
            if not void and c2.button("Void", key=f"sb_void_{r['invoice_no']}", use_container_width=True,
                                      help="Cancel the bill and put the cans back into stock"):
                billing.void_bill(r["invoice_no"])
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
    page_header("Scan & Bill", "Scan the cans, save the bill — stock goes down at once, and the camera knows who bought.")
    left, right = st.columns([5, 6], gap="large")
    with left:
        _scanner()
        _didnt_buy()
    with right:
        _bill()
    _today()
    _labels()
    html('<div class="apx-muted" style="margin-top:6px">Tip: open this screen on the counter phone and Store Camera on the '
         "laptop — the bill is matched to the customer the camera sees at the counter.</div>")
    nav.link("camera", "See who bought on Store Camera →")
