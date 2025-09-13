# app.py
# --------------------------------------------
# Рішення в умовах ризику — інтерактивне демо критеріїв:
#  - Бернуллі–Лаплас (EV)
#  - Гурвіц (alpha)
#  - Ходжес–Леман
#  - Мінімум дисперсії
#  - EV − γ·Var (mean–variance)
#  - Оцінювання ймовірностей MaxEnt (Гіббс–Джейнс)
# Dash ≥ 3: app.run(...) і ctx, allow_duplicate для duplicate outputs.
# --------------------------------------------

import io
import base64
import numpy as np
import pandas as pd
from dash import (
    Dash, dcc, html,
    Input, Output, State,
    dash_table, callback, ctx, no_update
)
import plotly.graph_objs as go

# ---------- Дані за замовчуванням ----------
DEFAULT_PAYOFF = pd.DataFrame(
    [[8, 6, 4],
     [2, 10, 1],
     [5, 7, 3]],
    columns=[f"S{j+1}" for j in range(3)],
    index=[f"A{i+1}" for i in range(3)]
)
DEFAULT_PROBS = pd.Series([0.3, 0.5, 0.2], index=DEFAULT_PAYOFF.columns)

# ---------- Утиліти ----------
def normalize_probs(p):
    p = np.array(p, dtype=float)
    p = np.where(np.isfinite(p) & (p >= 0.0), p, 0.0)
    s = p.sum()
    if s <= 0:
        return np.ones_like(p) / len(p)
    return p / s

def safe_float(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def rows_to_df(rows, columns):
    if not rows or not columns:
        return DEFAULT_PAYOFF.copy()
    cols_ids = [c["id"] for c in columns if c["id"] != "Стратегія"]
    if len(cols_ids) == 0:
        cols_ids = list(DEFAULT_PAYOFF.columns)
    idx, vals = [], []
    for i, r in enumerate(rows):
        idx.append(r.get("Стратегія", f"A{i+1}"))
        vals.append([safe_float(r.get(c, 0.0)) for c in cols_ids])
    df = pd.DataFrame(vals, columns=cols_ids, index=idx)
    if df.empty:
        df = DEFAULT_PAYOFF.copy()
    return df

# ---------- Критерії ----------
def bernoulli_laplace(payoff, probs):
    ev = payoff @ probs
    best_idx = int(np.argmax(ev))
    return ev, best_idx

def hurwicz(payoff, alpha=0.6):
    mins = payoff.min(axis=1)
    maxs = payoff.max(axis=1)
    val = alpha * maxs + (1.0 - alpha) * mins
    best_idx = int(np.argmax(val))
    return val, best_idx

def hodges_lehmann(payoff, probs):
    ev = payoff @ probs
    mins = payoff.min(axis=1)
    val = 0.5 * (ev + mins)
    best_idx = int(np.argmax(val))
    return val, best_idx

def variance_per_row(payoff, probs):
    ev = payoff @ probs
    ev2 = (payoff ** 2) @ probs
    var = ev2 - ev ** 2
    return var, ev

def min_variance(payoff, probs):
    var, ev = variance_per_row(payoff, probs)
    best_idx = int(np.argmin(var))
    return var, best_idx

def mean_variance(payoff, probs, gamma=0.1):
    var, ev = variance_per_row(payoff, probs)
    score = ev - gamma * var
    best_idx = int(np.argmax(score))
    return score, best_idx, ev, var

# ---------- MaxEnt (Гіббс–Джейнс) ----------
def maxent_probs(m, target, tol=1e-9, max_iter=200):
    m = np.asarray(m, dtype=float)
    if np.allclose(m, m[0]):
        return np.ones_like(m) / len(m)

    lo, hi = -1e6, 1e6

    def Z(lmbd):
        a = np.exp(np.clip(lmbd * m, -700, 700))
        return a.sum(), a

    def moment(lmbd):
        z, a = Z(lmbd)
        p = a / z
        return float((p * m).sum()), p

    t = float(np.clip(target, m.min(), m.max()))
    p_mid = np.ones_like(m) / len(m)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        mom, p_mid = moment(mid)
        if abs(mom - t) < tol:
            return p_mid
        if mom < t:
            lo = mid
        else:
            hi = mid
    return p_mid

# ---------- App ----------
app = Dash(__name__)
server = app.server

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "16px"},
    children=[
        html.H1("Критерії прийняття рішень в умовах ризику — Dash демо"),
        html.P("Редагуйте матрицю виграшів і ймовірності станів. Оберіть критерій і порівняйте результати. "
               "Можна оцінити ймовірності за принципом максимуму ентропії (Гіббс–Джейнс)."),

        html.Div(style={"display": "flex", "gap": "16px", "alignItems": "flex-start"}, children=[
            # --------- Ліва панель ----------
            html.Div(style={"flex": 1}, children=[
                html.H3("Матриця виграшів"),
                dash_table.DataTable(
                    id="payoff-table",
                    columns=([{"name": "Стратегія", "id": "Стратегія"}] +
                             [{"name": c, "id": c, "type": "numeric"} for c in DEFAULT_PAYOFF.columns]),
                    data=[dict({"Стратегія": idx},
                               **{c: float(DEFAULT_PAYOFF.loc[idx, c]) for c in DEFAULT_PAYOFF.columns})
                          for idx in DEFAULT_PAYOFF.index],
                    editable=True, row_deletable=True,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "center", "padding": "6px"},
                    style_header={"fontWeight": "700"},
                ),
                html.Div(style={"display": "flex", "gap": "8px", "marginTop": "8px"}, children=[
                    html.Button("Додати стратегію", id="add-row", n_clicks=0),
                    html.Button("Додати стан", id="add-col", n_clicks=0),
                    html.Button("Скинути до прикладу", id="reset", n_clicks=0),
                    html.Button("Експорт CSV", id="export", n_clicks=0),
                    dcc.Download(id="dl"),
                ]),
                html.Div(id="table-msg", style={"marginTop": "6px", "fontStyle": "italic"}),
            ]),
            # --------- Права панель ----------
            html.Div(style={"flex": 1}, children=[
                html.H3("Ймовірності станів"),
                html.Div(style={"display": "flex", "gap": "8px", "alignItems": "center"}, children=[
                    html.Label("Режим оцінювання p:"),
                    dcc.Dropdown(
                        id="prob-mode",
                        options=[
                            {"label": "Ручний (таблиця нижче)", "value": "manual"},
                            {"label": "MaxEnt (Гіббс–Джейнс)", "value": "maxent"},
                        ],
                        value="manual", clearable=False, style={"width": "280px"}
                    ),
                ]),
                dash_table.DataTable(
                    id="prob-table",
                    columns=[{"name": c, "id": c, "type": "numeric"} for c in DEFAULT_PAYOFF.columns],
                    data=[{c: float(DEFAULT_PROBS[c]) for c in DEFAULT_PAYOFF.columns}],
                    editable=True,
                    style_cell={"textAlign": "center", "padding": "6px"},
                    style_header={"fontWeight": "700"},
                ),

                html.Details(open=False, children=[
                    html.Summary("Налаштування MaxEnt (Гіббс–Джейнс)"),
                    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"}, children=[
                        html.Div(children=[
                            html.Label("Джерело m_j для обмеження"),
                            dcc.Dropdown(
                                id="maxent-source",
                                options=[
                                    {"label": "Стовпцеві середні (по станах)", "value": "colmean"},
                                    {"label": "Стратегія A1", "value": "A1"},
                                    {"label": "Стратегія A2", "value": "A2"},
                                    {"label": "Стратегія A3", "value": "A3"},
                                ],
                                value="colmean", clearable=False
                            ),
                        ]),
                        html.Div(children=[
                            html.Label("Цільове очікування M"),
                            dcc.Slider(id="maxent-target", min=0, max=10, step=0.1, value=6.0,
                                       marks=None, tooltip={"always_visible": True}),
                        ]),
                    ]),
                    html.Div(id="maxent-hint", style={"fontSize": "12px", "color": "#555", "marginTop": "4px"}),
                ], style={"marginTop": "8px"}),

                html.Div(style={"marginTop": "8px"}, children=[
                    html.Label("Критерій рішення"),
                    dcc.Dropdown(
                        id="criterion",
                        options=[
                            {"label": "Бернуллі–Лаплас (макс. EV)", "value": "BL"},
                            {"label": "Гурвіц (α-компроміс)", "value": "HUR"},
                            {"label": "Ходжес–Леман", "value": "HL"},
                            {"label": "Мінімум дисперсії", "value": "VARMIN"},
                            {"label": "EV − γ·Var (mean–variance)", "value": "MV"},
                        ],
                        value="BL", clearable=False
                    ),
                    html.Div(style={"display": "flex", "gap": "12px", "alignItems": "center", "marginTop": "8px"}, children=[
                        html.Label("α для Гурвіца"),
                        dcc.Slider(id="alpha", min=0.0, max=1.0, step=0.05, value=0.6,
                                   marks=None, tooltip={"always_visible": True}),
                    ]),
                    html.Div(style={"display": "flex", "gap": "12px", "alignItems": "center", "marginTop": "8px"}, children=[
                        html.Label("γ для EV−γ·Var"),
                        dcc.Slider(id="gamma", min=0.0, max=1.0, step=0.05, value=0.1,
                                   marks=None, tooltip={"always_visible": True}),
                    ]),
                ]),
                html.Div(id="summary", style={"marginTop": "12px", "fontWeight": "600"}),
            ]),
        ]),

        html.Hr(),

        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
            dcc.Graph(id="payoff-heatmap"),
            dcc.Graph(id="criterion-bars"),
        ]),
        dcc.Graph(id="hurwicz-sensitivity"),
    ],
)

# ---------- Callback: модифікація таблиць ----------
@callback(
    Output("payoff-table", "columns"),
    Output("payoff-table", "data"),
    Output("prob-table", "columns"),
    Output("prob-table", "data"),
    Output("table-msg", "children"),
    Input("add-row", "n_clicks"),
    Input("add-col", "n_clicks"),
    Input("reset", "n_clicks"),
    State("payoff-table", "columns"),
    State("payoff-table", "data"),
    State("prob-table", "columns"),
    State("prob-table", "data"),
    prevent_initial_call=True,
)
def modify_table(add_row, add_col, reset, p_cols, p_data, pr_cols, pr_data):
    trig = ctx.triggered_id
    payoff_df = rows_to_df(p_data, p_cols) if p_data and p_cols else DEFAULT_PAYOFF.copy()
    probs = pd.Series(pr_data[0], dtype=float) if pr_data and len(pr_data) > 0 else DEFAULT_PROBS.copy()

    msg = ""
    if trig == "reset":
        payoff_df = DEFAULT_PAYOFF.copy()
        probs = DEFAULT_PROBS.copy()
        msg = "Скинуто до прикладу."
    elif trig == "add-row":
        i = 1
        while True:
            cand = f"A{len(payoff_df.index) + i}"
            if cand not in payoff_df.index:
                payoff_df.loc[cand] = 0.0
                break
            i += 1
        msg = f"Додано стратегію {cand}."
    elif trig == "add-col":
        j = 1
        while True:
            cand = f"S{len(payoff_df.columns) + j}"
            if cand not in payoff_df.columns:
                payoff_df[cand] = 0.0
                break
            j += 1
        probs = probs.reindex(payoff_df.columns).fillna(0.0)
        probs[cand] = 1e-9
        msg = f"Додано стан {cand}."

    new_p_cols = (
        [{"name": "Стратегія", "id": "Стратегія"}] +
        [{"name": c, "id": c, "type": "numeric"} for c in payoff_df.columns]
    )
    new_p_data = [
        dict({"Стратегія": idx}, **{c: float(payoff_df.loc[idx, c]) for c in payoff_df.columns})
        for idx in payoff_df.index
    ]
    new_pr_cols = [{"name": c, "id": c, "type": "numeric"} for c in payoff_df.columns]
    norm_probs = normalize_probs([probs.get(c, 0.0) for c in payoff_df.columns])
    new_pr_data = [{c: float(norm_probs[i]) for i, c in enumerate(payoff_df.columns)}]

    return new_p_cols, new_p_data, new_pr_cols, new_pr_data, msg

# ---------- Callback: експорт CSV ----------
@callback(
    Output("dl", "data"),
    Input("export", "n_clicks"),
    State("payoff-table", "data"),
    State("payoff-table", "columns"),
    State("prob-table", "data"),
    State("criterion", "value"),
    State("alpha", "value"),
    State("gamma", "value"),
    prevent_initial_call=True
)
def export_csv(n, p_data, p_cols, pr_data, crit, alpha, gamma):
    if not n:
        return no_update
    df = rows_to_df(p_data, p_cols)
    cols = list(df.columns)
    probs = np.array([safe_float(pr_data[0].get(c, 0.0)) for c in cols]) if pr_data else np.ones(len(cols))/len(cols)
    probs = normalize_probs(probs)
    csv = io.StringIO()
    csv.write("# Матриця виграшів\n")
    df.to_csv(csv)
    csv.write("\n# Ймовірності станів (нормовані)\n")
    pd.Series(probs, index=cols).to_csv(csv, header=["p"])
    csv.write(f"\n# Обрані налаштування: criterion={crit}, alpha={alpha}, gamma={gamma}\n")
    content = "data:text/csv;charset=utf-8," + base64.b64encode(csv.getvalue().encode()).decode()
    return dict(content=content, filename="decision_demo_export.csv")

# ---------- Callback: графіки/підсумки + MaxEnt ----------
@callback(
    Output("payoff-heatmap", "figure"),
    Output("criterion-bars", "figure"),
    Output("hurwicz-sensitivity", "figure"),
    Output("summary", "children"),
    Output("prob-table", "data", allow_duplicate=True),  # <-- важливо: дозволяємо дублікати
    Output("maxent-hint", "children"),
    Input("payoff-table", "data"),
    Input("payoff-table", "columns"),
    Input("prob-table", "data"),
    Input("prob-mode", "value"),
    Input("maxent-source", "value"),
    Input("maxent-target", "value"),
    Input("criterion", "value"),
    Input("alpha", "value"),
    Input("gamma", "value"),
    prevent_initial_call=True  # щоб не дублювалося на старті
)
def update_graphs(p_data, p_cols, pr_data, prob_mode, maxent_source, maxent_target, crit, alpha, gamma):
    payoff_df = rows_to_df(p_data, p_cols)
    cols_ids = list(payoff_df.columns)
    idx = list(payoff_df.index)
    payoff = payoff_df.to_numpy(dtype=float)

    # --- оцінка ймовірностей ---
    manual_probs = normalize_probs([safe_float(pr_data[0].get(c, 0.0)) for c in cols_ids]) if pr_data else np.ones(len(cols_ids))/len(cols_ids)
    maxent_msg = ""
    probs = manual_probs.copy()

    if prob_mode == "maxent":
        if maxent_source == "colmean":
            m = payoff_df.mean(axis=0).to_numpy(dtype=float)
            m_label = "стовпцеві середні"
        elif maxent_source in payoff_df.index:
            m = payoff_df.loc[maxent_source].to_numpy(dtype=float)
            m_label = f"рядок {maxent_source}"
        else:
            m = payoff_df.mean(axis=0).to_numpy(dtype=float)
            m_label = "стовпцеві середні (fallback)"

        m_min, m_max = float(np.min(m)), float(np.max(m))
        target = float(np.clip(safe_float(maxent_target, (m_min + m_max)/2), m_min, m_max))
        probs = maxent_probs(m, target)
        maxent_msg = (f"MaxEnt: m_j = {m_label}. Діапазон M: [{m_min:.3g}, {m_max:.3g}]. "
                      f"Використано M = {target:.3g}. p_j ∝ exp(λ·m_j).")

    # --- критерії ---
    bl_vals, bl_idx = bernoulli_laplace(payoff, probs)
    hur_vals, hur_idx = hurwicz(payoff, alpha=float(alpha))
    hl_vals, hl_idx = hodges_lehmann(payoff, probs)
    var_vals, var_idx = min_variance(payoff, probs)
    mv_score, mv_idx, ev_mv, var_mv = mean_variance(payoff, probs, gamma=float(gamma))

    if     crit == "BL":     chosen_idx, chosen_name, plot_vals = bl_idx, "Бернуллі–Лаплас", bl_vals
    elif   crit == "HUR":    chosen_idx, chosen_name, plot_vals = hur_idx, f"Гурвіц (α={alpha:.2f})", hur_vals
    elif   crit == "HL":     chosen_idx, chosen_name, plot_vals = hl_idx, "Ходжес–Леман", hl_vals
    elif   crit == "VARMIN": chosen_idx, chosen_name, plot_vals = var_idx, "Мінімум дисперсії", -var_vals
    else:                    chosen_idx, chosen_name, plot_vals = mv_idx, f"EV−γ·Var (γ={gamma:.2f})", mv_score

    heat = go.Figure(data=go.Heatmap(z=payoff, x=cols_ids, y=idx, colorbar=dict(title="Виграш")))
    heat.update_layout(title="Теплокарта матриці виграшів")

    bars = go.Figure()
    bars.add_trace(go.Bar(name="EV (Бернуллі–Лаплас)", x=idx, y=bl_vals))
    bars.add_trace(go.Bar(name=f"Гурвіц α={alpha:.2f}", x=idx, y=hur_vals))
    bars.add_trace(go.Bar(name="Ходжес–Леман", x=idx, y=hl_vals))
    bars.add_trace(go.Bar(name="−Var (мін. дисперсія)", x=idx, y=-var_vals))
    bars.add_trace(go.Bar(name=f"EV−γ·Var γ={gamma:.2f}", x=idx, y=mv_score))
    bars.update_layout(barmode="group", title="Порівняння критеріїв")

    alphas = np.linspace(0, 1, 41)
    sens_vals = [hurwicz(payoff, a)[0] for a in alphas]
    fig_sens = go.Figure()
    for i, name in enumerate(idx):
        fig_sens.add_trace(go.Scatter(x=alphas, y=[v[i] for v in sens_vals], mode="lines", name=name))
    fig_sens.update_layout(title="Чутливість критерію Гурвіца до α", xaxis_title="α", yaxis_title="Значення")

    summary = (
        f"Обраний критерій: {chosen_name}. Рекомендована стратегія: {idx[chosen_idx]}. "
        f"EV найкращої за очікуванням = {bl_vals[bl_idx]:.4g}. "
        f"Мінімальна дисперсія у {idx[var_idx]} (Var={var_vals[var_idx]:.4g}). "
        f"Найкраща за EV−γ·Var (γ={gamma:.2f}) — {idx[mv_idx]} (EV={ev_mv[mv_idx]:.4g}, Var={var_mv[mv_idx]:.4g}). "
        f"Ймовірності станів ({'MaxEnt' if prob_mode=='maxent' else 'ручні, нормовані'}): "
        + ", ".join([f'{p:.3f}' for p in probs]) + "."
    )

    prob_row = [{c: float(probs[i]) for i, c in enumerate(cols_ids)}]
    return heat, bars, fig_sens, summary, prob_row, maxent_msg

# ---------- Запуск ----------
if __name__ == "__main__":
    app.run(debug=True, port=8050, use_reloader=False)
