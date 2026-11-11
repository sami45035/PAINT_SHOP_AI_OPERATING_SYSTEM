"""
Keep one problem from taking the whole app down.

Streamlit Cloud regularly loads the home page as a health check; if anything on
it raises, the app shows "Oh no. Error running app" to everyone. Every screen
(and the start-up steps) therefore runs through `guard()`: an error is shown in
place with its details, written to the server log and to data/live/errors.log,
and the rest of the app keeps working.
"""
from __future__ import annotations

import sys
import threading
import traceback
from collections import deque
from datetime import datetime
from typing import Any, Callable, Deque, Dict, Optional

RECENT: Deque[Dict[str, str]] = deque(maxlen=20)
_LOCK = threading.Lock()


def is_control_flow(exc: BaseException) -> bool:
    """st.rerun(), st.stop(), st.switch_page() work by raising — those are not errors."""
    names = {c.__name__ for c in type(exc).__mro__}
    return bool(names & {"ScriptControlException", "RerunException", "StopException", "SwitchPageException"}) \
        or "Rerun" in type(exc).__name__


def report(where: str, exc: BaseException) -> str:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[apex] ERROR in {where} at {stamp}\n{tb}", file=sys.stderr, flush=True)
    with _LOCK:
        RECENT.appendleft({"where": where, "when": stamp, "error": f"{type(exc).__name__}: {exc}", "trace": tb})
        try:
            from src.config import DB_PATH

            log = DB_PATH.parent / "errors.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            lines = log.read_text(encoding="utf-8").splitlines()[-400:] if log.exists() else []
            lines += [f"--- {stamp} · {where}", *tb.rstrip().splitlines()]
            log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:  # noqa: BLE001 - logging must never fail the app
            pass
    return tb


def guard(where: str, fn: Callable[..., Any], *args, show: bool = True, default: Any = None, **kwargs) -> Any:
    """Run fn; on an error show a short message (with details folded away) instead of crashing."""
    try:
        return fn(*args, **kwargs)
    except BaseException as exc:  # noqa: BLE001
        if is_control_flow(exc) or isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        tb = report(where, exc)
        if show:
            try:
                import streamlit as st

                st.error(f"Something went wrong in **{where}** ({type(exc).__name__}: {str(exc)[:160]}). "
                         "The rest of the app still works — try **Refresh data** in Settings.")
                with st.expander("Details for the developer"):
                    st.code(tb)
            except Exception:  # noqa: BLE001
                pass
        return default


def last_errors(n: int = 5):
    with _LOCK:
        return list(RECENT)[:n]


def once(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Run a set-up step at most once at a time across all sessions (threads) of the server."""
    lock = threading.Lock()

    def wrapper(*a, **k):
        with lock:
            return fn(*a, **k)

    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapper


__all__ = ["guard", "report", "last_errors", "is_control_flow", "once", "Optional"]
