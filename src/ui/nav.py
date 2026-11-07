"""Page registry so screens can link to each other (filled in by app.py)."""
import streamlit as st

PAGES = {}


def link(name: str, label: str, icon: str = None) -> None:
    page = PAGES.get(name)
    if page is None:
        return
    if icon:
        st.page_link(page, label=label, icon=icon)
    else:
        st.page_link(page, label=label)


def go(name: str) -> None:
    page = PAGES.get(name)
    if page is not None:
        st.switch_page(page)
