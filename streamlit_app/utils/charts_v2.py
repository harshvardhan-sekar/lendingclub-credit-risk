"""Reusable Plotly chart builders for the Streamlit credit risk dashboard."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from .styles import GRADE_COLORS, GRADE_ORDER, SCENARIO_COLORS, DPD_COLORS, RAG_COLORS


def _grade_color_seq():
    return [GRADE_COLORS.get(g, "#999") for g in GRADE_ORDER]


# ── Bar Charts ──

def bar_by_grade(df: pd.DataFrame, y_col: str, title: str,
                 y_label: str = "", pct: bool = False) -> go.Figure:
    """Horizontal or vertical bar chart colored by grade."""
    fig = px.bar(
        df.sort_values("grade"),
        x="grade", y=y_col,
        color="grade",
        color_discrete_map=GRADE_COLORS,
        title=title,
    )
    if pct:
        fig.update_yaxes(tickformat=".1%")
    fig.update_layout(
        showlegend=False,
        yaxis_title=y_label or y_col,
        xaxis_title="Grade",
        template="plotly_white",
        height=400,
    )
    return fig


def grouped_bar_scenarios(df: pd.DataFrame, y_col: str, title: str,
                          y_label: str = "") -> go.Figure:
    """Grouped bar chart by grade × scenario."""
    fig = px.bar(
        df, x="grade", y=y_col, color="scenario",
        barmode="group",
        color_discrete_map=SCENARIO_COLORS,
        title=title,
    )
    fig.update_layout(
        template="plotly_white", height=450,
        yaxis_title=y_label or y_col,
    )
    return fig


# ── Line Charts ──

def line_over_time(df: pd.DataFrame, x_col: str, y_col: str, title: str,
                   color_col: str = None, y_label: str = "",
                   pct: bool = False) -> go.Figure:
    """Time-series line chart, optionally colored by a category."""
    fig = px.line(
        df, x=x_col, y=y_col, color=color_col,
        color_discrete_map=GRADE_COLORS if color_col == "grade" else None,
        title=title,
    )
    if pct:
        fig.update_yaxes(tickformat=".2%")
    fig.update_layout(
        template="plotly_white", height=400,
        yaxis_title=y_label or y_col,
        xaxis_title="",
    )
    return fig


def multi_line_scenarios(df: pd.DataFrame, x_col: str, y_col: str,
                         title: str, y_label: str = "") -> go.Figure:
    """Line chart with one line per scenario."""
    fig = px.line(
        df, x=x_col, y=y_col, color="scenario",
        color_discrete_map=SCENARIO_COLORS,
        title=title,
        markers=True,
    )
    fig.update_layout(
        template="plotly_white", height=400,
        yaxis_title=y_label or y_col,
    )
    return fig


# ── Heatmap ──

def heatmap(df_pivot: pd.DataFrame, title: str, fmt: str = ".2f",
            colorscale: str = "RdYlGn_r", height: int = 500) -> go.Figure:
    """Generic heatmap from a pivoted DataFrame."""
    fig = go.Figure(data=go.Heatmap(
        z=df_pivot.values,
        x=[str(c) for c in df_pivot.columns],
        y=[str(r) for r in df_pivot.index],
        texttemplate=f"%{{z:{fmt}}}",
        colorscale=colorscale,
        showscale=True,
    ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
    )
    return fig


# ── Tornado / Waterfall ──

def tornado_chart(data: list[dict], title: str = "Sensitivity Tornado") -> go.Figure:
    """Tornado chart from sensitivity data with 'factor', 'low', 'high' keys."""
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No sensitivity data available", showarrow=False)
        return fig

    df = pd.DataFrame(data)
    if "factor" not in df.columns:
        # Try to adapt
        for col in ["Factor", "name", "variable"]:
            if col in df.columns:
                df = df.rename(columns={col: "factor"})
                break

    if "low" not in df.columns or "high" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Tornado data missing low/high columns", showarrow=False)
        return fig

    df["range"] = df["high"] - df["low"]
    df = df.sort_values("range", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["factor"], x=df["low"] - df.get("base", df["low"]),
        orientation="h", name="Low", marker_color="#3498db",
    ))
    fig.add_trace(go.Bar(
        y=df["factor"], x=df["high"] - df.get("base", df["low"]),
        orientation="h", name="High", marker_color="#e74c3c",
    ))
    fig.update_layout(
        barmode="overlay",
        title=title,
        template="plotly_white",
        height=400,
        xaxis_title="ECL Impact",
    )
    return fig


def ecl_waterfall(labels: list, values: list, title: str = "ECL Waterfall") -> go.Figure:
    """Waterfall chart for ECL components."""
    fig = go.Figure(go.Waterfall(
        name="ECL", orientation="v",
        x=labels, y=values,
        connector={"line": {"color": "#7f8c8d"}},
        increasing={"marker": {"color": "#e74c3c"}},
        decreasing={"marker": {"color": "#27ae60"}},
        totals={"marker": {"color": "#2c3e50"}},
    ))
    fig.update_layout(title=title, template="plotly_white", height=450)
    return fig


# ── Donut / Pie ──

def donut_chart(labels: list, values: list, title: str,
                colors: list = None) -> go.Figure:
    """Donut chart."""
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.5,
        marker_colors=colors,
        textinfo="label+percent",
    ))
    fig.update_layout(title=title, template="plotly_white", height=400)
    return fig


# ── RAG Status ──

def rag_status_styled(df: pd.DataFrame) -> str:
    """Return HTML table with RAG-colored status cells."""
    rows = []
    for _, row in df.iterrows():
        status = row.get("RAG_Status", "Green")
        color = RAG_COLORS.get(status, "#999")
        symbol = row.get("Symbol", "")
        val = row.get("Value", "")
        if isinstance(val, float):
            val = f"{val:.4f}"
        rows.append(f"""
        <tr>
            <td style="padding:8px;font-weight:600">{row['Metric']}</td>
            <td style="padding:8px;text-align:center">{val}</td>
            <td style="padding:8px;text-align:center;background:{color};color:white;
                border-radius:4px;font-weight:600">{symbol} {status}</td>
        </tr>""")

    return f"""
    <table style="width:100%;border-collapse:collapse;border:1px solid #dee2e6">
        <thead>
            <tr style="background:#2c3e50;color:white">
                <th style="padding:10px;text-align:left">Metric</th>
                <th style="padding:10px;text-align:center">Value</th>
                <th style="padding:10px;text-align:center">RAG Status</th>
            </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
    </table>"""
