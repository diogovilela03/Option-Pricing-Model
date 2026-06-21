"""Streamlit entry point. Run with: streamlit run dashboard/app.py"""
import streamlit as st

st.set_page_config(
    page_title="Option Pricing Lab",
    page_icon="📈",
    layout="wide",
)

st.title("Option Pricing Lab")

tab1, tab2 = st.tabs(["Manual Parameters", "Market Data Snapshot"])

from dashboard.tab1_manual import render_tab1
from dashboard.tab2_market import render_tab2

with tab1:
    render_tab1()

with tab2:
    render_tab2()
