"""
AI sales predictor (Layer 3): learns from the shop's own sales history.

For each product group (interior emulsion, exterior emulsion, primer, ...),
from the daily cans sold over the last six months (only days the shop has
records for):

- the weekday pattern (Saturday vs Tuesday): ridge regression on
  log(1 + cans) with weekday and festival indicators
- how much each festival / season in the calendar REALLY moves sales for that
  group: average weekday-adjusted cans on its days against plain days, pulled
  toward the calendar's guess when there are only a few days (the guess weighs
  like a week of evidence); a festival not yet seen in the history counts at
  half the guess
- the long-run trend, measured only on plain days (so a season at the end of
  the history isn't mistaken for growth), damped and capped

A forecast day is then: recent level × weekday factor × festival factor ×
damped trend. Each group's forecast is split across its products by their
recent share of the group's sales, and turned into rupees at each product's
price.

The model is judged honestly by a backtest: fit only on data before a cut-off,
forecast the next 4 weeks without seeing them, compare with what really sold,
and compare with the simple method (average of the last 30 days).
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.formatting import ratio
from src import clock
from src.config import DB_PATH
from src.predictor import event_applies

TRAIN_DAYS = 180
LEVEL_DAYS = 28            # the recent level is the last 4 weeks, with weekday/festival effects removed
SHARE_DAYS = 56
TREND_DAMP = 0.5
TREND_CAP = 0.15           # at most ±15% over the forecast horizon
RIDGE = 2.0
UNPROVEN_STRENGTH = 0.5    # a festival the history hasn't seen yet counts at half the calendar's guess
EFFECT_PRIOR_DAYS = 7      # the calendar's guess weighs like a week of evidence
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ----------------------------------------------------------------------------- data

def load_history(until: Optional[date] = None, days: int = TRAIN_DAYS + 40) -> Dict[str, pd.DataFrame]:
    """Daily cans per product (complete days before `until`), the product list and the season calendar."""
    until = until or clock.today()
    start = until - timedelta(days=days)
    with sqlite3.connect(DB_PATH) as conn:
        sales = pd.read_sql_query(
            "SELECT DATE(sale_timestamp) AS day, product_id, SUM(quantity) AS units, SUM(total_amount) AS revenue "
            "FROM sales WHERE DATE(sale_timestamp) >= ? AND DATE(sale_timestamp) < ? GROUP BY day, product_id;",
            conn, params=(start.isoformat(), until.isoformat()))
        products = pd.read_sql_query(
            "SELECT id AS product_id, sku, brand, shade_name, hex_code, category, finish, pack_size_litres, "
            "selling_price, cost_price FROM products;", conn)
        cal = pd.read_sql_query("SELECT * FROM seasonal_calendar;", conn)
    sales["day"] = pd.to_datetime(sales["day"]).dt.date
    cal["start"] = pd.to_datetime(cal["start_date"]).dt.date
    cal["end"] = pd.to_datetime(cal["end_date"]).dt.date
    return {"sales": sales, "products": products, "calendar": cal}


def group_of(category: str) -> str:
    return category or "Other"


# ----------------------------------------------------------------------------- model

@dataclass
class GroupModel:
    group: str
    events: List[Dict[str, Any]]              # relevant calendar events: name, start, end, prior
    dow_effect: np.ndarray                    # 7 multiplicative factors (mean 1 over a week)
    event_effect: Dict[str, float]            # learned multiplicative effect per event
    trend_per_day: float                      # damped, as a fraction per day
    level: float                              # cans per "plain" day (no festival, average weekday)
    fitted_days: int
    seen: Dict[str, int] = field(default_factory=dict)   # festival -> days of it in the history


@dataclass
class Forecaster:
    """Fit once (up to a cut-off day), then forecast any number of days after it."""
    cutoff: date
    groups: Dict[str, GroupModel] = field(default_factory=dict)
    shares: Dict[int, float] = field(default_factory=dict)       # product -> share of its group's cans
    products: Optional[pd.DataFrame] = None

    # ---- fitting
    @classmethod
    def fit(cls, data: Dict[str, pd.DataFrame], cutoff: date) -> "Forecaster":
        sales, products, cal = data["sales"], data["products"], data["calendar"]
        sales = sales[sales["day"] < cutoff]
        self = cls(cutoff=cutoff, products=products)
        # learn only from days the shop has records for: before its first sale is "unknown", not "sold nothing"
        # (zeros there would look like strong growth and distort the season effects)
        first = min(sales["day"]) if not sales.empty else cutoff - timedelta(days=1)
        span = max(14, min(TRAIN_DAYS, (cutoff - first).days))
        days = [cutoff - timedelta(days=i) for i in range(span, 0, -1)]
        prod_group = dict(zip(products["product_id"], products["category"].map(group_of)))
        sales = sales.assign(group=sales["product_id"].map(prod_group))
        for g, gp in products.groupby(products["category"].map(group_of)):
            daily = sales[sales["group"] == g].groupby("day")["units"].sum()
            y = np.array([float(daily.get(d, 0.0)) for d in days])
            # festivals that touch this group (any product in it)
            evs = []
            for _, ev in cal.iterrows():
                if any(event_applies(r.category, r.finish, ev["impact_category"]) for r in gp.itertuples()):
                    evs.append({"name": ev["event_name"], "start": ev["start"], "end": ev["end"],
                                "prior": float(ev["expected_multiplier"])})
            self.groups[g] = _fit_group(g, days, y, evs)
            # product shares inside the group, smoothed so a product with no recent sale isn't forecast at zero
            recent = sales[(sales["group"] == g) & (sales["day"] >= cutoff - timedelta(days=SHARE_DAYS))]
            per = recent.groupby("product_id")["units"].sum()
            alpha = 0.5
            total = float(per.sum()) + alpha * len(gp)
            for pid in gp["product_id"]:
                self.shares[int(pid)] = (float(per.get(pid, 0.0)) + alpha) / total
        return self

    # ---- forecasting
    def group_daily(self, g: str, days: List[date]) -> np.ndarray:
        m = self.groups[g]
        out = []
        for d in days:
            h = (d - self.cutoff).days + 1
            trend = 1.0 + max(-TREND_CAP, min(TREND_CAP, m.trend_per_day * h))
            out.append(m.level * m.dow_effect[d.weekday()] * _event_factor(m, d) * trend)
        return np.array(out)

    def forecast(self, days: int = 30, start: Optional[date] = None) -> pd.DataFrame:
        """Cans and rupees per product per day."""
        start = start or self.cutoff
        dates = [start + timedelta(days=i) for i in range(days)]
        rows = []
        prods = self.products.set_index("product_id")
        for g in self.groups:
            gd = self.group_daily(g, dates)
            for pid, r in prods[prods["category"].map(group_of) == g].iterrows():
                s = self.shares.get(int(pid), 0.0)
                for d, u in zip(dates, gd):
                    rows.append((d, int(pid), u * s, u * s * float(r["selling_price"])))
        return pd.DataFrame(rows, columns=["day", "product_id", "units", "revenue"])

    def explain(self, product_id: int, days: int = 30) -> Dict[str, Any]:
        """Plain-language reasons behind one product's forecast."""
        p = self.products.set_index("product_id").loc[product_id]
        g = group_of(p["category"])
        m = self.groups[g]
        dates = [self.cutoff + timedelta(days=i) for i in range(days)]
        active = [(e["name"], m.event_effect[e["name"]], e["end"]) for e in m.events
                  if any(e["start"] <= d <= e["end"] for d in dates) and abs(m.event_effect[e["name"]] - 1) > 0.03]
        best, worst = int(np.argmax(m.dow_effect)), int(np.argmin(m.dow_effect))
        return {
            "group": g, "share_pct": round(self.shares.get(int(product_id), 0) * 100, 1),
            "events": [{"name": n, "effect": round(f, 2), "until": e} for n, f, e in active],
            "best_day": WEEKDAYS[best], "best_day_factor": round(float(m.dow_effect[best]), 2),
            "worst_day": WEEKDAYS[worst], "worst_day_factor": round(float(m.dow_effect[worst]), 2),
            "trend_30d_pct": round(max(-TREND_CAP, min(TREND_CAP, m.trend_per_day * 30)) * 100),
        }


def _event_factor(m: GroupModel, d: date) -> float:
    ups, downs = [], []
    for e in m.events:
        if e["start"] <= d <= e["end"]:
            f = m.event_effect[e["name"]]
            (ups if f >= 1 else downs).append(f)
    return (max(ups) if ups else 1.0) * (min(downs) if downs else 1.0)


def _fit_group(g: str, days: List[date], y: np.ndarray, events: List[Dict[str, Any]]) -> GroupModel:
    """Ridge regression on log(1 + cans): weekday dummies and festival indicators; then the trend on plain days."""
    n = len(days)
    t = np.arange(n) / max(1, n - 1)
    cols, names = [np.ones(n)], ["intercept"]
    for k in range(1, 7):                                     # Monday is the reference day
        cols.append(np.array([1.0 if d.weekday() == k else 0.0 for d in days]))
        names.append(f"dow{k}")
    usable = []
    for e in events:
        a = np.array([1.0 if e["start"] <= d <= e["end"] else 0.0 for d in days])
        if a.sum() >= 3:                                      # enough days to learn from
            cols.append(a)
            names.append(e["name"])
            usable.append(e)
    X = np.column_stack(cols)
    z = np.log1p(y)
    # ridge toward priors: weekday 0, trend 0, festival = the calendar's multiplier (in log(1 + cans) terms)
    prior = np.zeros(X.shape[1])
    ybar = max(float(y.mean()), 1e-6)
    for i, nm in enumerate(names):
        e = next((e for e in usable if e["name"] == nm), None)
        if e is not None:
            prior[i] = math.log1p(ybar * e["prior"]) - math.log1p(ybar)
    penalty = np.full(X.shape[1], RIDGE)
    penalty[0] = 0.0
    A = X.T @ X + np.diag(penalty)
    b = X.T @ z + penalty * prior
    beta = np.linalg.solve(A, b)

    # back to multiplicative factors on cans: compare predicted cans with/without each effect
    def cans(xrow):
        return max(0.0, math.expm1(float(xrow @ beta)))

    base_row = np.zeros(X.shape[1])
    base_row[0] = 1.0
    base = max(cans(base_row), 1e-6)
    dow = np.ones(7)
    for k in range(1, 7):
        r = base_row.copy()
        r[k] = 1.0
        dow[k] = cans(r) / base
    dow = dow / dow.mean()
    ev_eff = {}
    # Festival effects on cans, measured directly: average weekday-adjusted cans on the festival's days
    # against plain days (no festival running), pulled toward the calendar's guess when there are few days.
    # (The log regression above gives the weekday pattern; read off it, a festival's effect on a small
    # group — a few cans a day, some days none — comes out too big.)
    adj = np.array([v / dow[d.weekday()] for d, v in zip(days, y)])
    plain = np.array([not any(e2["start"] <= d <= e2["end"] for e2 in events) for d in days])
    plain_mean = float(adj[plain].mean()) if plain.sum() >= 14 else float(adj.mean())
    for e in events:
        if e in usable:
            on = np.array([e["start"] <= d <= e["end"] for d in days])
            n_on = int(on.sum())
            raw = float(adj[on].mean()) / max(plain_mean, 1e-6)
            w = n_on / (n_on + EFFECT_PRIOR_DAYS)
            eff = math.exp(w * math.log(max(raw, 1e-3)) + (1 - w) * math.log(e["prior"]))
            ev_eff[e["name"]] = float(min(3.0, max(0.2, eff)))
        else:
            # Not in the history yet: the calendar's number is only a guess, so it counts at half strength
            # (2.2x becomes about 1.5x) until the shop's own sales confirm it.
            ev_eff[e["name"]] = float(math.exp(UNPROVEN_STRENGTH * math.log(e["prior"])))
    # Trend, measured separately and only on days with no festival or season running: when a season
    # sits at the end of the history, "season on" and "growing" look alike in one regression, and the
    # trend would swallow the season (then keep "growing" after the season ends).
    plain_days = np.array([not any(e["start"] <= d <= e["end"] for e in events) for d in days])
    resid = z - X @ beta
    slope = 0.0
    if plain_days.sum() >= 42:
        tt = np.arange(n, dtype=float)[plain_days]
        slope = float(np.polyfit(tt, resid[plain_days], 1)[0])       # log cans per day
    trend = TREND_DAMP * slope
    seen = {e["name"]: int(sum(1 for d in days if e["start"] <= d <= e["end"])) for e in events}
    m = GroupModel(g, events, dow, ev_eff, trend, 0.0, n, seen)
    # recent level: last LEVEL_DAYS with weekday and festival effects divided out
    recent = [(d, v) for d, v in zip(days, y)][-LEVEL_DAYS:]
    plain = [v / (dow[d.weekday()] * _event_factor(m, d)) for d, v in recent]
    m.level = float(np.mean(plain)) if plain else 0.0
    return m


# ----------------------------------------------------------------------------- simple baseline

def simple_average_forecast(data: Dict[str, pd.DataFrame], cutoff: date, days: int = 28) -> pd.DataFrame:
    """The usual shop method: average cans a day over the last 30 days, every day the same."""
    s = data["sales"]
    recent = s[(s["day"] < cutoff) & (s["day"] >= cutoff - timedelta(days=30))]
    rate = recent.groupby("product_id")["units"].sum() / 30.0
    price = data["products"].set_index("product_id")["selling_price"]
    dates = [cutoff + timedelta(days=i) for i in range(days)]
    rows = [(d, int(pid), float(r), float(r) * float(price[pid])) for pid, r in rate.items() for d in dates]
    return pd.DataFrame(rows, columns=["day", "product_id", "units", "revenue"])


# ----------------------------------------------------------------------------- backtest

def backtest(horizon: int = 28, until: Optional[date] = None) -> Dict[str, Any]:
    """
    Pretend it's `horizon` days ago: fit on what was known then, forecast the
    next `horizon` days, and compare with what actually sold.
    """
    until = until or clock.today()
    cutoff = until - timedelta(days=horizon)
    data = load_history(until)
    fc = Forecaster.fit(data, cutoff)
    pred = fc.forecast(horizon, cutoff)
    base = simple_average_forecast(data, cutoff, horizon)
    actual = data["sales"][(data["sales"]["day"] >= cutoff) & (data["sales"]["day"] < until)]
    dates = [cutoff + timedelta(days=i) for i in range(horizon)]

    def daily(df, col="revenue"):
        s = df.groupby("day")[col].sum()
        return np.array([float(s.get(d, 0.0)) for d in dates])

    a, p, b = daily(actual), daily(pred), daily(base)

    def weekly(x):
        return np.array([x[i:i + 7].sum() for i in range(0, horizon - horizon % 7, 7)])

    def wape(x, y):
        return float(np.abs(x - y).sum() / max(1e-9, x.sum()) * 100)

    aw, pw, bw = weekly(a), weekly(p), weekly(b)
    # product level: total cans over the horizon, for products that sold at least 5
    au = actual.groupby("product_id")["units"].sum()
    pu = pred.groupby("product_id")["units"].sum()
    bu = base.groupby("product_id")["units"].sum()
    sold = au[au >= 5]
    prod_err = float(np.abs(sold - pu.reindex(sold.index).fillna(0)).sum() / max(1, sold.sum()) * 100)
    prod_err_b = float(np.abs(sold - bu.reindex(sold.index).fillna(0)).sum() / max(1, sold.sum()) * 100)
    ratio = a / np.maximum(p, 1e-9)
    ratio = ratio[np.isfinite(ratio) & (p > 0)]
    return {
        "cutoff": cutoff, "until": until, "dates": dates, "actual": a, "predicted": p, "simple": b,
        "total_actual": float(a.sum()), "total_predicted": float(p.sum()), "total_simple": float(b.sum()),
        "total_error_pct": float((p.sum() - a.sum()) / max(1e-9, a.sum()) * 100),
        "total_error_simple_pct": float((b.sum() - a.sum()) / max(1e-9, a.sum()) * 100),
        "weekly_error_pct": wape(aw, pw), "weekly_error_simple_pct": wape(aw, bw),
        "daily_error_pct": wape(a, p), "daily_error_simple_pct": wape(a, b),
        "product_error_pct": prod_err, "product_error_simple_pct": prod_err_b,
        "products_checked": int(len(sold)),
        # spread of real days around the forecast -> the likely range shown on the chart
        "band_low": float(np.quantile(ratio, 0.1)) if len(ratio) else 0.6,
        "band_high": float(np.quantile(ratio, 0.9)) if len(ratio) else 1.4,
        "weekly_band_low": float(np.min(aw / np.maximum(pw, 1e-9))) if len(aw) else 0.8,
        "weekly_band_high": float(np.max(aw / np.maximum(pw, 1e-9))) if len(aw) else 1.2,
    }


def rolling_backtest(horizon: int = 14, windows: int = 10, until: Optional[date] = None) -> Dict[str, Any]:
    """
    The same test repeated: for each of the last `windows` weeks, forecast the
    next `horizon` days from what was known then. Errors are on cans (prices
    differ by pack, cans are what the shop orders).
    """
    until = until or clock.today()
    data = load_history(until, days=TRAIN_DAYS + horizon + 7 * windows + 40)
    cat = data["products"].set_index("product_id")["category"]
    rows = []
    for k in range(windows):
        cutoff = until - timedelta(days=horizon + 7 * k)
        pred = Forecaster.fit(data, cutoff).forecast(horizon, cutoff)
        base = simple_average_forecast(data, cutoff, horizon)
        s = data["sales"]
        act = s[(s["day"] >= cutoff) & (s["day"] < cutoff + timedelta(days=horizon))]
        ta, tp, tb = act["units"].sum(), pred["units"].sum(), base["units"].sum()
        A = act.groupby(act["product_id"].map(cat))["units"].sum()
        P = pred.groupby(pred["product_id"].map(cat))["units"].sum().reindex(A.index).fillna(0)
        B = base.groupby(base["product_id"].map(cat))["units"].sum().reindex(A.index).fillna(0)
        rows.append({
            "from": cutoff, "to": cutoff + timedelta(days=horizon - 1), "actual": float(ta),
            "ai": float(tp), "simple": float(tb),
            "ai_err_pct": float(abs(tp - ta) / max(ta, 1e-9) * 100), "simple_err_pct": float(abs(tb - ta) / max(ta, 1e-9) * 100),
            "ai_group_err_pct": float((A - P).abs().sum() / max(A.sum(), 1e-9) * 100),
            "simple_group_err_pct": float((A - B).abs().sum() / max(A.sum(), 1e-9) * 100),
        })
    df = pd.DataFrame(rows).sort_values("from").reset_index(drop=True)
    return {
        "periods": df, "horizon": horizon,
        "ai_err_pct": float(df["ai_err_pct"].mean()), "simple_err_pct": float(df["simple_err_pct"].mean()),
        "ai_group_err_pct": float(df["ai_group_err_pct"].mean()),
        "simple_group_err_pct": float(df["simple_group_err_pct"].mean()),
        "ai_wins": int((df["ai_err_pct"] < df["simple_err_pct"]).sum()), "windows": len(df),
        "ai_group_wins": int((df["ai_group_err_pct"] < df["simple_group_err_pct"]).sum()),
    }


# ----------------------------------------------------------------------------- what the screen needs

def outlook(days: int = 30, today: Optional[date] = None) -> Dict[str, Any]:
    """Fit on all complete days up to today; forecast today onwards."""
    today = today or clock.today()
    data = load_history(today)
    fc = Forecaster.fit(data, today)
    pred = fc.forecast(days, today)
    prods = data["products"].set_index("product_id")
    hist = data["sales"]
    last30 = hist[hist["day"] >= today - timedelta(days=30)].groupby("product_id")["units"].sum()
    per = pred.groupby("product_id")[["units", "revenue"]].sum()
    per = per.join(prods[["brand", "shade_name", "hex_code", "category", "pack_size_litres", "selling_price"]])
    per["last30"] = last30.reindex(per.index).fillna(0)
    per["change_pct"] = ratio(per["units"] - per["last30"], per["last30"], min_den=2.5, digits=0)
    per["label"] = per["shade_name"] + " " + per["pack_size_litres"].map(lambda x: f"{x:g} L")
    per = per.sort_values("revenue", ascending=False)
    daily = pred.groupby("day")[["units", "revenue"]].sum().reset_index()
    hist_daily = hist[hist["day"] >= today - timedelta(days=60)].groupby("day")[["units", "revenue"]].sum().reset_index()
    events = []
    for g, m in fc.groups.items():
        for e in m.events:
            if e["end"] >= today - timedelta(days=60) and e["start"] <= today + timedelta(days=days):
                events.append({"name": e["name"], "start": e["start"], "end": e["end"], "group": g,
                               "learned": round(m.event_effect[e["name"]], 2), "calendar": e["prior"],
                               "days_seen": m.seen.get(e["name"], 0)})
    dow = {g: m.dow_effect for g, m in fc.groups.items()}
    # shop-wide weekday pattern: each group's pattern weighted by its size
    w = np.array([m.level for m in fc.groups.values()])
    shop_dow = (np.array([m.dow_effect for m in fc.groups.values()]) * w[:, None]).sum(0) / max(w.sum(), 1e-9)
    return {"model": fc, "daily": daily, "history": hist_daily, "products": per.reset_index(), "events": events,
            "dow": dow, "shop_dow": (shop_dow / shop_dow.mean()).tolist(), "today": today,
            "learned": learned_effects(fc)}


def learned_effects(fc: "Forecaster") -> List[Dict[str, Any]]:
    """Every festival the model knows, per product group: calendar guess vs what the shop's sales show."""
    rows = []
    for g, m in fc.groups.items():
        for e in m.events:
            rows.append({"festival": e["name"], "group": g, "start": e["start"], "end": e["end"],
                         "calendar": e["prior"], "learned": round(m.event_effect[e["name"]], 2),
                         "days_seen": m.seen.get(e["name"], 0)})
    return rows


def product_history(product_id: int, days: int = 60, today: Optional[date] = None) -> pd.DataFrame:
    today = today or clock.today()
    start = today - timedelta(days=days)
    with sqlite3.connect(DB_PATH) as conn:
        h = pd.read_sql_query(
            "SELECT DATE(sale_timestamp) AS day, SUM(quantity) AS units, SUM(total_amount) AS revenue FROM sales "
            "WHERE product_id = ? AND DATE(sale_timestamp) >= ? AND DATE(sale_timestamp) < ? GROUP BY day;",
            conn, params=(int(product_id), start.isoformat(), today.isoformat()))
    h["day"] = pd.to_datetime(h["day"]).dt.date
    full = pd.DataFrame({"day": [start + timedelta(days=i) for i in range(days)]})
    return full.merge(h, on="day", how="left").fillna(0)
