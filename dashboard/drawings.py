"""Your own lines on each stock's chart: drawn with the chart's drawing tools (or a price level typed in), saved to
data/drawings/<SYM>.json and drawn again on the next visit. Display only: never read by the strategy."""
import json
import os
import shutil

import streamlit as st
import streamlit.components.v1 as components

from dashboard.settings import DRAWINGS_DIR

_DIR = os.path.join(os.path.dirname(__file__), "chart_component")
_JS = os.path.join(_DIR, "plotly.min.js")
if not os.path.exists(_JS):                      # plotly.js from the installed plotly package (offline, same version)
    import plotly
    shutil.copy(os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js"), _JS)
_chart = components.declare_component("sa_stock_chart", path=_DIR)
LEVEL = dict(color="#7c3aed", width=1.5, dash="dash")
TOOLS = ["drawline", "drawopenpath", "drawrect", "eraseshape"]

def _path(sym):
    return os.path.join(DRAWINGS_DIR, f"{''.join(c for c in sym if c.isalnum() or c in '.-')}.json")

def load(sym):
    try:
        with open(_path(sym)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []

def save(sym, shapes):
    os.makedirs(DRAWINGS_DIR, exist_ok=True)
    tmp = _path(sym) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(shapes, f, indent=1)
    os.replace(tmp, _path(sym))                   # never a half-written file

def level(price):
    """A horizontal line at `price` across the whole price panel."""
    return dict(type="line", xref="x domain", x0=0, x1=1, yref="y", y0=price, y1=price, line=LEVEL, editable=True,
                label=dict(text=f"${price:,.2f}", textposition="end", font=dict(size=10, color=LEVEL["color"])))

def render_chart(fig, sym, ohlc):
    """The chart with your saved shapes; a drawn / moved / erased shape is saved at once."""
    for s in fig.layout.shapes:
        s.name = "sys"                            # the chart's own shapes (earnings lines, held periods) are never saved
    shapes = load(sym)
    fig.update_layout(shapes=list(fig.layout.shapes) + shapes, dragmode=False,   # range only via the 1M..All buttons
                      newshape=dict(line=dict(color="#7c3aed", width=2), fillcolor="rgba(124,58,237,0.08)"))
    spec = json.loads(fig.to_json())
    spec["config"] = {"displaylogo": False, "scrollZoom": False, "modeBarButtonsToAdd": TOOLS,
                      "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "zoom2d", "pan2d", "zoomIn2d", "zoomOut2d"]}
    out = _chart(spec=json.dumps(spec), ohlc=ohlc, key=f"chart_{sym}", default=None)
    if out and out.get("t") != st.session_state.get(f"_drawn_{sym}"):
        st.session_state[f"_drawn_{sym}"] = out["t"]
        if out["shapes"] != shapes:
            save(sym, out["shapes"])
            st.rerun()
    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    px = c1.number_input("Price level", min_value=0.0, value=None, step=1.0, format="%.2f", key=f"lvl_{sym}",
                         placeholder="Add a horizontal price level…", label_visibility="collapsed")
    if c2.button("Add level", key=f"lvl_add_{sym}", width="stretch", disabled=not px):
        save(sym, shapes + [level(px)])
        st.rerun()
    if c3.button(f"Clear drawings ({len(shapes)})", key=f"clr_{sym}", width="stretch", disabled=not shapes):
        save(sym, [])
        st.rerun()
