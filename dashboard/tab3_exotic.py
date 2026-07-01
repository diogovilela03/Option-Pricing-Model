"""Tab 3 — Exotic Structures: entry point routing to sub-renderers."""
import streamlit as st

from dashboard.tab3_strategies import (
    render_strategies_sidebar, render_strategies,
)
from dashboard.tab3_path_dep import (
    render_path_dep_sidebar, render_path_dep,
)
from dashboard.tab3_autocall import render_autocall_sidebar, render_autocall
from dashboard.tab3_multi import render_multi_sidebar, render_multi

CATEGORIES = [
    "Option Strategies",
    "Path-Dependent",
    "Structured Products",
    "Autocallables",
    "Multi-Asset",
]

_CATEGORY_DESC = {
    "Option Strategies":   "P&L diagrams for spreads, straddles, strangles, butterflies and more.",
    "Path-Dependent":      "Digital, Asian, barrier, double-barrier and quanto options.",
    "Structured Products": "Reverse convertibles, discount/bonus/airbag/twin-win certificates.",
    "Autocallables":       "Autocall incremental, Phoenix, and Phoenix Memory (Monte Carlo).",
    "Multi-Asset":         "Basket, worst-of and rainbow options on correlated assets.",
}


def render_tab3_sidebar() -> dict:
    """Collect Tab 3 sidebar inputs. Uses st.sidebar.* so safe to call outside any context."""
    category = st.sidebar.selectbox("Category", CATEGORIES)
    st.sidebar.caption(_CATEGORY_DESC[category])
    st.sidebar.markdown("---")

    if category == "Option Strategies":
        params = render_strategies_sidebar()
    elif category in ("Path-Dependent", "Structured Products"):
        params = render_path_dep_sidebar(category)
    elif category == "Autocallables":
        params = render_autocall_sidebar()
    else:
        params = render_multi_sidebar()

    params["category"] = category
    return params


def render_tab3_content(params: dict):
    """Render main Tab 3 content. Call inside `with tab3:` context."""
    category = params["category"]
    if category == "Option Strategies":
        render_strategies(params)
    elif category in ("Path-Dependent", "Structured Products"):
        render_path_dep(params)
    elif category == "Autocallables":
        render_autocall(params)
    else:
        render_multi(params)
