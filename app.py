"""
Интерактивный дашборд СППР: мультивалютный портфель
для сохранения покупательной способности.

Расчёт выполняет модуль sppr_core.py (тот же, что в ноутбуке) по локальным данным
из папки data/, поэтому интернет для запуска не нужен.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

import sppr_core as core
from sppr_core import CASH, PROFILE_LABELS, PROFILES

print("Расчёт модели...")
MODEL = core.run_model()
PRICES_REAL = MODEL["market"]["prices_real"]
print("Модель готова.")

app = Dash(__name__)
app.title = "Мультивалютный портфель"

CARD_STYLE = {
    "background": "#f7f7f5",
    "border": "1px solid #ddd",
    "borderRadius": "8px",
    "padding": "12px 14px",
    "minWidth": "150px",
}
TABLE_STYLE = {
    "style_table": {"overflowX": "auto"},
    "style_cell": {"textAlign": "center", "padding": "6px", "fontFamily": "Segoe UI, Arial, sans-serif"},
    "style_header": {"fontWeight": "bold"},
}
MARGIN = dict(l=40, r=20, t=50, b=40)
FEATURE_NAMES = {
    "volatility": "Волатильность",
    "drawdown": "Макс. годовая просадка",
    "preservation_probability": "Вероятность сохранения",
    "liquidity": "Ликвидность",
}


def card(title: str, value: str) -> html.Div:
    return html.Div([html.Div(title), html.H3(value, style={"margin": "4px 0 0"})], style=CARD_STYLE)


def table_payload(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    return df.to_dict("records"), [{"name": c, "id": c} for c in df.columns]


app.layout = html.Div(
    style={"fontFamily": "Segoe UI, Arial, sans-serif", "margin": "20px", "maxWidth": "1200px"},
    children=[
        html.H2("Мультивалютный портфель для сохранения покупательной способности"),
        html.P(
            "Система поддержки принятия решений: рекомендательный отбор инструментов, "
            "паритет риска и выбор точки на границе «просадка — сохранность». "
            f"Ограничения: номинальная годовая просадка ≤ {core.DRAWDOWN_LIMIT:.0%}, "
            f"доля нерублёвых инструментов ≥ {core.MIN_NON_CASH_SHARE:.0%}. "
            f"Доходность RUB_CASH: ключевая ставка ЦБ − {core.CASH_SPREAD:g} п.п."
        ),
        html.Label("Профиль инвестора"),
        dcc.Dropdown(
            id="profile",
            options=[
                {"label": f"{spec['label']} (горизонт {spec['horizon']} мес.)", "value": key}
                for key, spec in PROFILES.items()
            ],
            value="liquidity_reserve",
            clearable=False,
            style={"maxWidth": "460px"},
        ),
        html.Div(id="metrics", style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginTop": "16px"}),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginTop": "16px"},
            children=[dcc.Graph(id="weights-chart"), dcc.Graph(id="frontier-chart")],
        ),
        dcc.Graph(id="path-chart"),
        html.H4("Рекомендательный отбор"),
        dash_table.DataTable(id="scores-table", **TABLE_STYLE),
        html.H4("Стресс-тестирование"),
        dash_table.DataTable(id="stress-table", **TABLE_STYLE),
        html.H4("Сверка паритета риска с Excel Solver (профиль «Резерв ликвидности»)"),
        html.P(id="solver-status"),
        dash_table.DataTable(id="solver-table", **TABLE_STYLE),
    ],
)


@app.callback(
    Output("metrics", "children"),
    Output("weights-chart", "figure"),
    Output("frontier-chart", "figure"),
    Output("path-chart", "figure"),
    Output("scores-table", "data"),
    Output("scores-table", "columns"),
    Output("stress-table", "data"),
    Output("stress-table", "columns"),
    Input("profile", "value"),
)
def update_dashboard(profile: str):
    item = MODEL["profiles"][profile]
    best = item["best"]
    weights = item["weights"]
    frontier = item["frontier"]
    horizon = PROFILES[profile]["horizon"]

    metrics = [
        card(f"Сохранность за {horizon} мес.", f"{best['preservation']:.3f}"),
        card("Вероятность сохранения", f"{best['preservation_probability']:.0%}"),
        card("Номинальная просадка", f"{best['nominal_drawdown']:.1%}"),
        card("Реальная просадка", f"{best['real_drawdown']:.1%}"),
        card("Доля нерублёвых", f"{best['non_cash_share']:.0%}"),
        card("Долларовая экспозиция", f"{item['exposure']['USD']:.0%}"),
        html.Div(
            [html.Div("Статус"), html.P(item["status"], style={"margin": 0, "fontSize": "13px"})],
            style={**CARD_STYLE, "flex": "1"},
        ),
    ]

    weights_fig = go.Figure(
        data=[
            go.Bar(
                x=weights.index.tolist(),
                y=weights.values,
                marker_color=["#8c8c8c" if a == CASH else "#2f5d8c" for a in weights.index],
                text=[f"{v:.0%}" for v in weights.values],
                textposition="outside",
            )
        ]
    )
    weights_fig.update_layout(
        title="Рекомендуемые веса портфеля",
        yaxis_title="Доля",
        yaxis_tickformat=".0%",
        margin=MARGIN,
    )

    allowed = frontier[frontier["non_cash_share"] >= core.MIN_NON_CASH_SHARE - 1e-9]
    frontier_fig = go.Figure()
    frontier_fig.add_trace(
        go.Scatter(
            x=frontier["nominal_drawdown"] * 100,
            y=frontier["preservation"],
            mode="lines",
            name="Все точки смешения",
            line=dict(color="#9bb3cc"),
            customdata=frontier["mixing_coefficient"],
            hovertemplate="m = %{customdata:.2f}<br>просадка %{x:.1f}%<br>сохранность %{y:.3f}",
        )
    )
    frontier_fig.add_trace(
        go.Scatter(
            x=allowed["nominal_drawdown"] * 100,
            y=allowed["preservation"],
            mode="lines",
            name="Доля нерублёвых ≥ 50%",
            line=dict(color="#2f5d8c", width=4),
            customdata=allowed["mixing_coefficient"],
            hovertemplate="m = %{customdata:.2f}<br>просадка %{x:.1f}%<br>сохранность %{y:.3f}",
        )
    )
    frontier_fig.add_vline(x=core.DRAWDOWN_LIMIT * 100, line_dash="dash", line_color="red")
    frontier_fig.add_trace(
        go.Scatter(
            x=[best["nominal_drawdown"] * 100],
            y=[best["preservation"]],
            mode="markers",
            marker=dict(size=14, color="orange", line=dict(color="black", width=1)),
            name="Рекомендация",
        )
    )
    frontier_fig.update_layout(
        title="Граница «номинальная просадка — реальная сохранность»",
        xaxis_title="Номинальная годовая просадка, %",
        yaxis_title=f"Средний реальный множитель за {horizon} мес.",
        margin=MARGIN,
        legend=dict(orientation="h", y=-0.25),
    )

    path = core.portfolio_path(weights, PRICES_REAL)
    path_fig = go.Figure()
    path_fig.add_trace(
        go.Scatter(x=path.index, y=path.values, mode="lines", name="Рекомендованный портфель", line=dict(color="#1b7f5a"))
    )
    path_fig.add_trace(
        go.Scatter(
            x=PRICES_REAL.index,
            y=PRICES_REAL[CASH],
            mode="lines",
            name="Только RUB_CASH",
            line=dict(color="#8c8c8c", dash="dot"),
        )
    )
    path_fig.add_hline(y=1.0, line_dash="dot", line_color="black")
    path_fig.update_layout(
        title="Динамика реальной стоимости (конец января 2015 = 1)",
        yaxis_title="Индекс",
        margin=MARGIN,
    )

    scores = item["scores"].sort_values(ascending=False)
    features = item["features"].loc[scores.index]
    scores_df = pd.DataFrame(
        {
            "Инструмент": scores.index,
            "Интегральная оценка": scores.values.round(3),
            **{FEATURE_NAMES[c]: features[c].round(3).values for c in FEATURE_NAMES},
            "Отобран": ["да" if name in item["selected"].index else "нет" for name in scores.index],
        }
    )

    stress = item["stress"].copy()
    for column in ["Номинальный результат месяца", "Реальный результат месяца"]:
        stress[column] = stress[column].map(lambda v: "—" if pd.isna(v) else f"{v:+.1%}")
    column = "Номинальная просадка за 12 мес. с шоком"
    stress[column] = stress[column].map(lambda v: f"{v:.1%}")

    return (
        metrics,
        weights_fig,
        frontier_fig,
        path_fig,
        *table_payload(scores_df),
        *table_payload(stress),
    )


@app.callback(
    Output("solver-status", "children"),
    Output("solver-table", "data"),
    Output("solver-table", "columns"),
    Input("profile", "value"),
)
def update_solver(_profile: str):
    rp = MODEL["profiles"]["liquidity_reserve"]["risk_parity"]
    comparison, status = core.solver_comparison(rp["weights"], rp["covariance"])
    if comparison is None:
        comparison = pd.DataFrame({"Python": rp["weights"]})
        status += " Порядок действий — на листе «Инструкция» в data/solver_rp.xlsx."
    comparison = comparison.round(4).reset_index(names="Инструмент")
    return (status, *table_payload(comparison))


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
