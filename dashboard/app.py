"""Streamlit entry point. Run with: streamlit run dashboard/app.py"""
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of how Streamlit is launched
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from storage.db import load_chain

st.set_page_config(
    page_title="Option Pricing Lab",
    page_icon="📈",
    layout="wide",
)

st.title("Option Pricing Lab")

from dashboard.tab1_manual import render_tab1, render_tab1_sidebar
from dashboard.tab2_market import render_tab2, render_tab2_sidebar, _load
from dashboard.tab3_exotic import render_tab3_sidebar, render_tab3_content

DB_PATH = Path("data/spy_chain.db")

tab1, tab2, tab3 = st.tabs([
    "Manual Parameters",
    "Market Data Snapshot",
    "Exotic Structures",
])

with st.sidebar:
    active_tab = st.radio(
        "Active tab",
        ["Manual Parameters", "Market Data Snapshot", "Exotic Structures"],
        label_visibility="collapsed",
    )
    st.divider()

if active_tab == "Manual Parameters":
    with st.sidebar:
        t1_params = render_tab1_sidebar()
    with tab1:
        render_tab1(t1_params)
    with tab2:
        st.info("Switch to **Manual Parameters** in the sidebar to use this view.")
    with tab3:
        st.info("Switch to **Exotic Structures** in the sidebar to use this view.")

elif active_tab == "Market Data Snapshot":
    expiries = []
    if DB_PATH.exists():
        df = _load()
        expiries = sorted(df["expiration"].unique().tolist())
    with st.sidebar:
        t2_params = render_tab2_sidebar(expiries)
    with tab1:
        st.info("Switch to **Market Data Snapshot** in the sidebar to use this view.")
    with tab2:
        render_tab2(t2_params)
    with tab3:
        st.info("Switch to **Exotic Structures** in the sidebar to use this view.")

else:  # Exotic Structures
    # Sidebar inputs — st.sidebar.* calls route correctly regardless of context
    t3_params = render_tab3_sidebar()
    with tab1:
        st.info("Switch to **Exotic Structures** in the sidebar to use this view.")
    with tab2:
        st.info("Switch to **Exotic Structures** in the sidebar to use this view.")
    with tab3:
        render_tab3_content(t3_params)
