"""
One clock for the whole app.

Every "today", "now" and "is the shop open?" question goes through here, in the
store's own time zone (India by default). Streamlit Cloud servers run on UTC, so
using datetime.now() directly made the shop look "open" at 10:57 PM IST.
"""
from datetime import date, datetime, time, timedelta, timezone

from src.config import STORE_TIMEZONE, STORE_OPEN_HOUR, STORE_CLOSE_HOUR

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        STORE_TZ = ZoneInfo(STORE_TIMEZONE)
    except ZoneInfoNotFoundError:
        # Windows without the `tzdata` package has no zone database.
        STORE_TZ = timezone(timedelta(hours=5, minutes=30), "IST")
except ImportError:  # pragma: no cover - Python < 3.9
    STORE_TZ = timezone(timedelta(hours=5, minutes=30), "IST")


def now() -> datetime:
    """Current time in the store's time zone (timezone-aware)."""
    return datetime.now(STORE_TZ)


def now_naive() -> datetime:
    """Current store time without tzinfo — the format the SQLite tables use."""
    return now().replace(tzinfo=None)


def today() -> date:
    """Today's date in the store's time zone."""
    return now().date()


def today_str() -> str:
    return today().strftime("%Y-%m-%d")


def is_store_open(at: datetime | None = None) -> bool:
    at = at or now()
    return STORE_OPEN_HOUR <= at.hour < STORE_CLOSE_HOUR


def opening_label(at: datetime | None = None) -> str:
    """Short status line for the sidebar, e.g. 'Open · closes 9:00 PM'."""
    at = at or now()
    close_txt = fmt_time(datetime.combine(date.today(), time(STORE_CLOSE_HOUR % 24)))
    open_txt = fmt_time(datetime.combine(date.today(), time(STORE_OPEN_HOUR % 24)))
    if is_store_open(at):
        return f"Open · closes {close_txt}"
    return f"Closed · opens {open_txt}"


def fmt_time(dt: datetime) -> str:
    """'11:42 PM' on every OS (Windows strftime has no %-I)."""
    return dt.strftime("%I:%M %p").lstrip("0")


def fmt_day(d: date) -> str:
    """'Sat 7 Nov'"""
    return f"{d.strftime('%a')} {d.day} {d.strftime('%b')}"
