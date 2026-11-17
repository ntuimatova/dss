"""
Расчётное ядро СППР «Мультивалютный портфель для сохранения покупательной способности».

Модуль используется и в ноутбуке project.ipynb, и в дашборде app.py,
поэтому вся логика расчёта находится в одном месте.

Основные соглашения
-------------------
* Данные месячные. Строка с датой ``YYYY-MM-01`` — значение на конец этого месяца
  (цена закрытия последнего торгового дня месяца).
* Базовая точка — конец января 2015 года: все индексы в ней равны 1.
* «Номинальные» величины — в рублях, «реальные» — в рублях, очищенных от инфляции (ИПЦ РФ).
* ``RUB_CASH`` — рублёвые денежные средства (вклад / фонд денежного рынка), на которые
  начисляется доход по ключевой ставке ЦБ РФ за вычетом спреда ``CASH_SPREAD``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Пути и параметры
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PRICES_PATH = DATA_DIR / "prices_monthly.csv"
CPI_PATH = DATA_DIR / "cpi_ru.csv"
KEY_RATE_PATH = DATA_DIR / "key_rate_ru.csv"
SOLVER_PATH = DATA_DIR / "solver_rp.xlsx"

# Тикеры Yahoo Finance: USD/RUB, EUR/RUB, USD/CNY, золото (USD за унцию), ETF BIL и SHY
TICKERS = ["RUB=X", "EURRUB=X", "CNY=X", "GC=F", "BIL", "SHY"]
DOWNLOAD_START = "2014-12-01"
DOWNLOAD_END = "2026-11-01"  # не включительно: берём только завершённые месяцы

ANALYSIS_START = "2015-01-01"  # базовая точка — конец января 2015
ANALYSIS_END = "2026-10-01"  # последний месяц, по которому есть ИПЦ

RISKY_ASSETS = ["USD", "EUR", "CNY", "GOLD", "BIL", "SHY"]
CASH = "RUB_CASH"
ALL_ASSETS = RISKY_ASSETS + [CASH]

# Доходность рублёвого кеша = ключевая ставка − CASH_SPREAD (п.п. годовых).
# 1 п.п. — консервативная поправка на то, что ставки вкладов и доходность фондов
# денежного рынка (после комиссий) обычно ниже ключевой ставки.
CASH_SPREAD = 1.0

MAX_WEIGHT = 0.40  # максимальная доля одного нерублёвого инструмента
MIN_NON_CASH_SHARE = 0.50  # минимальная доля нерублёвых инструментов
DRAWDOWN_LIMIT = 0.22  # лимит номинальной максимальной годовой просадки
DRAWDOWN_WINDOW = 12  # «годовая» просадка: окно 12 месячных доходностей

LIQUIDITY = pd.Series(
    {
        "USD": 1.00,
        "EUR": 0.95,
        "CNY": 0.70,
        "GOLD": 0.85,
        "BIL": 0.90,
        "SHY": 0.90,
        "RUB_CASH": 1.00,
    }
)

CURRENCY_OF = {
    "USD": "USD",
    "BIL": "USD",
    "SHY": "USD",
    "EUR": "EUR",
    "CNY": "CNY",
    "GOLD": "Золото",
    "RUB_CASH": "RUB",
}

PROFILES = {
    "liquidity_reserve": {
        "label": "Резерв ликвидности",
        "horizon": 6,
        "feature_weights": {
            "low_volatility": 0.40,
            "low_drawdown": 0.25,
            "preservation_probability": 0.10,
            "liquidity": 0.25,
        },
    },
    "target_accumulation": {
        "label": "Целевое накопление",
        "horizon": 12,
        "feature_weights": {
            "low_volatility": 0.20,
            "low_drawdown": 0.45,
            "preservation_probability": 0.20,
            "liquidity": 0.15,
        },
    },
    "capital_preservation": {
        "label": "Сохранение капитала",
        "horizon": 48,
        "feature_weights": {
            "low_volatility": 0.15,
            "low_drawdown": 0.20,
            "preservation_probability": 0.50,
            "liquidity": 0.15,
        },
    },
}
PROFILE_LABELS = {key: value["label"] for key, value in PROFILES.items()}

# Стресс-сценарии: мгновенное изменение рублёвых цен инструментов за один месяц
# ("nominal") и, при необходимости, инфляция за этот месяц ("inflation").
STRESS_SCENARIOS = {
    "Ослабление USD к рублю на 20%": {
        "nominal": {"USD": -0.20, "BIL": -0.20, "SHY": -0.20, "GOLD": -0.20},
    },
    "Ослабление EUR к рублю на 20%": {"nominal": {"EUR": -0.20}},
    "Ослабление CNY к рублю на 20%": {"nominal": {"CNY": -0.20}},
    "Совместный шок рисковых активов": {
        "nominal": {
            "GOLD": -0.15,
            "USD": -0.10,
            "EUR": -0.10,
            "CNY": -0.10,
            "BIL": -0.03,
            "SHY": -0.03,
        },
    },
    "Инфляционный шок (+7,6% за месяц, как в марте 2022)": {
        "nominal": {},
        "inflation": 0.076,
    },
}


# ---------------------------------------------------------------------------
# 1. Загрузка данных
# ---------------------------------------------------------------------------


def download_monthly_prices() -> pd.DataFrame:
    """Скачивает дневные котировки из Yahoo Finance и берёт закрытие последнего дня месяца.

    Месячные бары yfinance для фьючерса на золото содержат пропуски, поэтому
    месячный ряд строится из дневных данных.
    """
    import yfinance as yf

    daily = yf.download(
        TICKERS,
        start=DOWNLOAD_START,
        end=DOWNLOAD_END,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )["Close"][TICKERS]
    monthly = daily.resample("ME").last()
    monthly.index = monthly.index.to_period("M").to_timestamp()
    monthly.index.name = "date"
    return monthly


def load_monthly_prices(refresh: bool = False) -> pd.DataFrame:
    """Месячные котировки из локального файла; при refresh=True — заново из Yahoo Finance."""
    if refresh or not PRICES_PATH.exists():
        prices = download_monthly_prices()
        prices.to_csv(PRICES_PATH, float_format="%.10g")
    return pd.read_csv(PRICES_PATH, parse_dates=["date"], index_col="date")


def _chain_index(monthly_rate: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Индекс, равный 1 в первой дате index и растущий на monthly_rate в каждом следующем месяце."""
    growth = (1 + monthly_rate.reindex(index)).cumprod()
    return growth / growth.iloc[0]


def load_cpi_index(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Помесячная инфляция и индекс уровня цен (1 в базовой точке)."""
    cpi = pd.read_csv(CPI_PATH, parse_dates=["date"], index_col="date")
    cpi = cpi.reindex(index)
    if cpi["inflation_mom"].isna().any():
        missing = list(cpi.index[cpi["inflation_mom"].isna()].strftime("%Y-%m"))
        raise ValueError(f"нет данных ИПЦ за месяцы: {missing}")
    cpi["cpi_index"] = _chain_index(cpi["inflation_mom"] / 100, index)
    return cpi


def load_key_rate_monthly(index: pd.DatetimeIndex) -> pd.Series:
    """Средняя за месяц ключевая ставка ЦБ РФ, % годовых (с учётом дат вступления в силу)."""
    decisions = pd.read_csv(KEY_RATE_PATH, parse_dates=["date"], index_col="date")["key_rate"]
    last_day = index.max() + pd.offsets.MonthEnd(0)
    daily = decisions.reindex(pd.date_range(decisions.index.min(), last_day, freq="D")).ffill()
    monthly = daily.resample("MS").mean()
    return monthly.reindex(index)


def cash_index(index: pd.DatetimeIndex, cash_spread: float | None = CASH_SPREAD) -> tuple[pd.Series, pd.Series]:
    """Индекс стоимости рублёвого кеша и его месячная доходность.

    cash_spread=None — кеш без процентов (допущение первоначальной версии работы).
    """
    if cash_spread is None:
        monthly_rate = pd.Series(0.0, index=index)
    else:
        key_rate = load_key_rate_monthly(index)
        monthly_rate = (key_rate - cash_spread).clip(lower=0) / 100 / 12
    return _chain_index(monthly_rate, index), monthly_rate


def to_rub_prices(raw: pd.DataFrame, cash: pd.Series) -> pd.DataFrame:
    """Перевод котировок в рубли."""
    usd_rub = raw["RUB=X"]
    return pd.DataFrame(
        {
            "USD": usd_rub,
            "EUR": raw["EURRUB=X"],
            "CNY": usd_rub / raw["CNY=X"],
            "GOLD": raw["GC=F"] * usd_rub,
            "BIL": raw["BIL"] * usd_rub,
            "SHY": raw["SHY"] * usd_rub,
            "RUB_CASH": cash,
        },
        index=raw.index,
    )


def load_market_data(cash_spread: float | None = CASH_SPREAD, refresh: bool = False) -> dict:
    """Все рыночные данные модели: номинальные и реальные рублёвые цены, ИПЦ, доходность кеша."""
    raw = load_monthly_prices(refresh=refresh).loc[ANALYSIS_START:ANALYSIS_END].dropna()
    index = raw.index
    cash, cash_rate = cash_index(index, cash_spread)
    prices_nominal = to_rub_prices(raw, cash)
    prices_nominal = prices_nominal / prices_nominal.iloc[0]
    cpi = load_cpi_index(index)
    prices_real = prices_nominal.div(cpi["cpi_index"], axis=0)
    return {
        "raw": raw,
        "prices_nominal": prices_nominal,
        "prices_real": prices_real,
        "cpi": cpi,
        "cash_rate": cash_rate,
    }


# ---------------------------------------------------------------------------
# 2. Показатели пути стоимости
# ---------------------------------------------------------------------------


def portfolio_path(weights: pd.Series, prices: pd.DataFrame) -> pd.Series:
    """Стоимость портфеля с ежемесячной ребалансировкой; в базовой точке равна 1.

    Первая доходность (февраль 2015) учитывается: путь начинается с 1 в январе.
    """
    returns = prices[list(weights.index)].pct_change().fillna(0.0)
    return (1 + returns @ weights).cumprod()


def rolling_max_drawdown(path: pd.Series, window: int = DRAWDOWN_WINDOW) -> float:
    """Наибольшая просадка внутри скользящего окна из `window` месячных доходностей (≥ 0)."""
    values = np.asarray(path, dtype=float)
    if len(values) <= window:
        segments = [values]
    else:
        segments = (values[end - window : end + 1] for end in range(window, len(values)))
    worst = 0.0
    for segment in segments:
        drawdown = (segment / np.maximum.accumulate(segment) - 1).min()
        worst = min(worst, drawdown)
    return abs(float(worst))


def horizon_multipliers(path: pd.Series, horizon: int) -> pd.Series:
    """Во сколько раз изменилась стоимость за каждые `horizon` месяцев (скользящие окна)."""
    values = np.asarray(path, dtype=float)
    return pd.Series(values[horizon:] / values[:-horizon], index=path.index[horizon:])


def preservation_stats(real_path: pd.Series, horizon: int) -> dict:
    """Показатели сохранности покупательной способности на горизонте профиля."""
    multipliers = horizon_multipliers(real_path, horizon)
    return {
        "preservation": float(multipliers.mean()),
        "preservation_probability": float((multipliers >= 1).mean()),
        "worst_horizon": float(multipliers.min()),
        "full_period": float(real_path.iloc[-1] / real_path.iloc[0]),
    }


# ---------------------------------------------------------------------------
# 3. Рекомендательный отбор
# ---------------------------------------------------------------------------


def compute_features(prices_real: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Признаки инструментов (в реальном выражении). RUB_CASH — справочно, в отборе не участвует."""
    returns = prices_real.pct_change().dropna()
    rows = {}
    for asset in prices_real.columns:
        multipliers = horizon_multipliers(prices_real[asset], horizon)
        rows[asset] = {
            "volatility": float(returns[asset].std() * np.sqrt(12)),
            "drawdown": rolling_max_drawdown(prices_real[asset]),
            "preservation_probability": float((multipliers >= 1).mean()),
            "liquidity": float(LIQUIDITY[asset]),
        }
    return pd.DataFrame(rows).T


def normalize_higher_better(series: pd.Series) -> pd.Series:
    spread = series.max() - series.min()
    if spread < 1e-12:
        return pd.Series(1.0, index=series.index)
    return (series - series.min()) / spread


def normalize_lower_better(series: pd.Series) -> pd.Series:
    return 1 - normalize_higher_better(series)


def normalize_features(features: pd.DataFrame) -> pd.DataFrame:
    """Min-max нормировка только по нерублёвым инструментам (кеш — заведомый выброс)."""
    risky = features.loc[RISKY_ASSETS]
    return pd.DataFrame(
        {
            "low_volatility": normalize_lower_better(risky["volatility"]),
            "low_drawdown": normalize_lower_better(risky["drawdown"]),
            "preservation_probability": normalize_higher_better(risky["preservation_probability"]),
            "liquidity": normalize_higher_better(risky["liquidity"]),
        }
    )


def integral_score(features_normalized: pd.DataFrame, feature_weights: dict) -> pd.Series:
    return sum(features_normalized[name] * weight for name, weight in feature_weights.items())


def select_instruments(scores: pd.Series, min_count: int = 4, threshold: float = 0.5) -> pd.Series:
    """Инструменты с оценкой ≥ threshold, но не меньше min_count лучших."""
    ranked = scores.sort_values(ascending=False)
    chosen = ranked[ranked >= threshold]
    if len(chosen) < min_count:
        chosen = ranked.head(min_count)
    return chosen


# ---------------------------------------------------------------------------
# 4. Паритет риска
# ---------------------------------------------------------------------------


def risk_contribution_shares(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    contributions = weights * (covariance @ weights)
    return contributions / contributions.sum()


def risk_parity_weights(returns: pd.DataFrame, max_weight: float = MAX_WEIGHT) -> dict:
    """Веса с равными вкладами в риск (SLSQP). Строится только по нерублёвым инструментам:
    у кеша почти нулевая дисперсия, и уравнять его вклад с остальными невозможно."""
    covariance = returns.cov().to_numpy() * 12
    n_assets = covariance.shape[0]
    target = 1.0 / n_assets

    def objective(weights: np.ndarray) -> float:
        return float(np.sum((risk_contribution_shares(weights, covariance) - target) ** 2))

    result = minimize(
        objective,
        np.full(n_assets, target),
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n_assets,
        constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},),
        options={"maxiter": 1000, "ftol": 1e-15},
    )
    if not result.success:
        raise RuntimeError(result.message)
    weights = pd.Series(result.x, index=returns.columns)
    return {
        "weights": weights,
        "risk_contribution": pd.Series(risk_contribution_shares(result.x, covariance), index=returns.columns),
        "volatility": float(np.sqrt(result.x @ covariance @ result.x)),
        "covariance": pd.DataFrame(covariance, index=returns.columns, columns=returns.columns),
        "objective": float(result.fun),
    }


# ---------------------------------------------------------------------------
# 5. Граница «просадка — сохранность»
# ---------------------------------------------------------------------------


def build_frontier(
    risk_parity: pd.Series,
    prices_nominal: pd.DataFrame,
    prices_real: pd.DataFrame,
    horizon: int,
    steps: int = 101,
) -> pd.DataFrame:
    """Смешение портфеля паритета риска (доля m) с рублёвым кешем (доля 1 − m)."""
    instruments = list(risk_parity.index) + [CASH]
    rp = risk_parity.reindex(instruments).fillna(0.0)
    cash = pd.Series(0.0, index=instruments)
    cash[CASH] = 1.0
    rows = []
    for mix in np.round(np.linspace(0, 1, steps), 4):
        weights = mix * rp + (1 - mix) * cash
        real_path = portfolio_path(weights, prices_real)
        nominal_path = portfolio_path(weights, prices_nominal)
        rows.append(
            {
                "mixing_coefficient": mix,
                **preservation_stats(real_path, horizon),
                "nominal_drawdown": rolling_max_drawdown(nominal_path),
                "real_drawdown": rolling_max_drawdown(real_path),
                "non_cash_share": float(1 - weights[CASH]),
                "max_single_weight": float(weights.drop(CASH).max()),
                "weights": weights,
            }
        )
    return pd.DataFrame(rows)


def choose_portfolio(
    frontier: pd.DataFrame,
    drawdown_limit: float = DRAWDOWN_LIMIT,
    min_non_cash: float = MIN_NON_CASH_SHARE,
) -> tuple[pd.Series, str, bool]:
    """Точка с максимальной сохранностью среди допустимых. Возвращает (строка, статус, допустимо ли)."""
    eps = 1e-9
    share_ok = frontier["non_cash_share"] >= min_non_cash - eps
    feasible = frontier[share_ok & (frontier["nominal_drawdown"] <= drawdown_limit + eps)]
    if len(feasible):
        best = feasible.loc[feasible["preservation"].idxmax()]
        status = (
            f"номинальная просадка ≤ {drawdown_limit:.0%} "
            f"и доля нерублёвых инструментов ≥ {min_non_cash:.0%}"
        )
        return best, status, True
    candidates = frontier[share_ok] if share_ok.any() else frontier
    best = candidates.loc[candidates["nominal_drawdown"].idxmin()]
    status = (
        "совместные ограничения недостижимы; выбрана минимальная номинальная просадка "
        f"при доле нерублёвых ≥ {min_non_cash:.0%}"
    )
    return best, status, False


def currency_exposure(weights: pd.Series) -> pd.Series:
    """Доли портфеля по валютам риска (BIL и SHY — долларовые инструменты)."""
    exposure = weights.groupby(lambda asset: CURRENCY_OF[asset]).sum()
    order = ["USD", "EUR", "CNY", "Золото", "RUB"]
    return exposure.reindex(order).fillna(0.0)


# ---------------------------------------------------------------------------
# 6. Стресс-тестирование
# ---------------------------------------------------------------------------


def stress_test(
    weights: pd.Series,
    prices_nominal: pd.DataFrame,
    prices_real: pd.DataFrame,
    scenarios: dict = STRESS_SCENARIOS,
    drawdown_limit: float = DRAWDOWN_LIMIT,
) -> pd.DataFrame:
    """Результат шокового месяца и просадка за последние 12 месяцев с учётом шока.

    Шоковый месяц добавляется после последней исторической даты; просадка считается
    по окну «последние 11 исторических месяцев + месяц шока» и сравнивается с лимитом.
    """
    nominal_path = portfolio_path(weights, prices_nominal)
    recent = nominal_path.iloc[-DRAWDOWN_WINDOW:]
    column = "Номинальная просадка за 12 мес. с шоком"
    rows = [
        {
            "Сценарий": "Без шока (последние 12 мес.)",
            "Номинальный результат месяца": np.nan,
            "Реальный результат месяца": np.nan,
            column: rolling_max_drawdown(nominal_path.iloc[-DRAWDOWN_WINDOW - 1 :]),
        }
    ]
    for name, scenario in scenarios.items():
        shock = pd.Series(scenario.get("nominal", {}), dtype=float).reindex(weights.index).fillna(0.0)
        nominal_return = float(shock @ weights)
        real_return = (1 + nominal_return) / (1 + scenario.get("inflation", 0.0)) - 1
        shocked = np.append(recent.to_numpy(), recent.iloc[-1] * (1 + nominal_return))
        rows.append(
            {
                "Сценарий": name,
                "Номинальный результат месяца": nominal_return,
                "Реальный результат месяца": real_return,
                column: rolling_max_drawdown(pd.Series(shocked)),
            }
        )
    table = pd.DataFrame(rows)
    table["Лимит просадки соблюдён"] = np.where(table[column] <= drawdown_limit + 1e-9, "да", "нет")
    return table


# ---------------------------------------------------------------------------
# 7. Полный расчёт модели
# ---------------------------------------------------------------------------


def run_model(
    cash_spread: float | None = CASH_SPREAD,
    drawdown_limit: float = DRAWDOWN_LIMIT,
    min_non_cash: float = MIN_NON_CASH_SHARE,
    market: dict | None = None,
) -> dict:
    """Весь конвейер СППР для всех профилей."""
    if market is None:
        market = load_market_data(cash_spread=cash_spread)
    prices_nominal = market["prices_nominal"]
    prices_real = market["prices_real"]
    returns_nominal = prices_nominal.pct_change().dropna()

    result = {"market": market, "profiles": {}}
    for profile, spec in PROFILES.items():
        features = compute_features(prices_real, spec["horizon"])
        normalized = normalize_features(features)
        scores = integral_score(normalized, spec["feature_weights"])
        selected = select_instruments(scores)
        rp = risk_parity_weights(returns_nominal[list(selected.index)])
        frontier = build_frontier(rp["weights"], prices_nominal, prices_real, spec["horizon"])
        best, status, feasible = choose_portfolio(frontier, drawdown_limit, min_non_cash)
        weights = best["weights"]
        result["profiles"][profile] = {
            "features": features,
            "features_normalized": normalized,
            "scores": scores,
            "selected": selected,
            "risk_parity": rp,
            "frontier": frontier,
            "best": best,
            "status": status,
            "feasible": feasible,
            "weights": weights,
            "exposure": currency_exposure(weights),
            "stress": stress_test(weights, prices_nominal, prices_real, drawdown_limit=drawdown_limit),
        }
    return result


def summary_table(model: dict) -> pd.DataFrame:
    """Сводка рекомендованных портфелей по профилям."""
    rows = {}
    for profile, item in model["profiles"].items():
        best = item["best"]
        rows[PROFILE_LABELS[profile]] = {
            "Горизонт, мес.": PROFILES[profile]["horizon"],
            "Доля паритета риска (m)": best["mixing_coefficient"],
            "Доля RUB_CASH": best["weights"][CASH],
            "Сохранность (средний реальный множитель)": best["preservation"],
            "Вероятность сохранения": best["preservation_probability"],
            "Худший результат на горизонте": best["worst_horizon"],
            "Реальная стоимость за весь период": best["full_period"],
            "Номинальная годовая просадка": best["nominal_drawdown"],
            "Реальная годовая просадка": best["real_drawdown"],
            "Ограничения выполнены": "да" if item["feasible"] else "нет",
        }
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Сверка с Excel Solver
# ---------------------------------------------------------------------------


def write_solver_workbook(
    covariance: pd.DataFrame,
    python_weights: pd.Series,
    path: Path = SOLVER_PATH,
    max_weight: float = MAX_WEIGHT,
) -> Path:
    """Книга Excel с моделью паритета риска для Solver.

    Лист «Модель»: веса (изменяемые ячейки, стартуют с 1/n), вклады в риск формулами,
    целевая ячейка — сумма квадратов отклонений долей вклада от 1/n.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    assets = list(covariance.index)
    n = len(assets)
    last = n + 1  # последняя строка с инструментами (строка 1 — заголовок)
    col = lambda j: chr(ord("B") + j)  # noqa: E731 — столбцы матрицы ковариации B, C, ...

    wb = Workbook()
    cov_ws = wb.active
    cov_ws.title = "Ковариация"
    cov_ws.append(["Годовая ковариация доходностей в рублях"] + assets)
    for asset in assets:
        cov_ws.append([asset] + [float(covariance.loc[asset, other]) for other in assets])

    ws = wb.create_sheet("Модель")
    header = [
        "Инструмент",
        "Вес (Solver)",
        "Предельный риск (Σw)",
        "Вклад в дисперсию",
        "Доля вклада",
        "Квадрат отклонения от 1/n",
        "Вес Python",
        "Разница",
    ]
    ws.append(header)
    for i, asset in enumerate(assets):
        r = i + 2
        marginal = "+".join(f"Ковариация!{col(j)}{r}*$B${j + 2}" for j in range(n))
        ws.append(
            [
                asset,
                round(1.0 / n, 6),
                f"={marginal}",
                f"=B{r}*C{r}",
                f"=D{r}/$D${last + 1}",
                f"=(E{r}-1/{n})^2",
                float(python_weights[asset]),
                f"=B{r}-G{r}",
            ]
        )
    total = last + 1
    ws.append(["Итого", f"=SUM(B2:B{last})", None, f"=SUM(D2:D{last})", f"=SUM(E2:E{last})", f"=SUM(F2:F{last})"])
    ws.append([])
    ws.append(["Целевая ячейка (минимизировать)", f"=F{total}*10000"])
    ws.append(["Волатильность портфеля", f"=SQRT(D{total})"])
    ws.append(["Макс. |разница| с Python", f"=MAX(MAX(H2:H{last}),-MIN(H2:H{last}))"])
    ws.append(["Ограничение на вес", max_weight])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for r in range(2, last + 1):
        ws[f"B{r}"].fill = PatternFill("solid", fgColor="FFF2CC")
    ws[f"B{total + 2}"].fill = PatternFill("solid", fgColor="DDEBF7")
    ws.column_dimensions["A"].width = 34
    for letter in "BCDEFGH":
        ws.column_dimensions[letter].width = 20

    guide = wb.create_sheet("Инструкция")
    for line in [
        "Сверка паритета риска в Excel Solver",
        "",
        "1. Данные → Поиск решения (Solver). Если пункта нет: Файл → Параметры → Надстройки → «Поиск решения».",
        f"2. Оптимизировать целевую функцию: Модель!$B${total + 2}, «Минимум».",
        f"3. Изменяя ячейки переменных: Модель!$B$2:$B${last} (жёлтые).",
        f"4. Ограничения: Модель!$B${total} = 1;  Модель!$B$2:$B${last} <= {max_weight};  "
        "флажок «Сделать переменные без ограничений неотрицательными».",
        "5. Метод решения: «Поиск решения нелинейных задач методом ОПГ» (GRG Nonlinear). "
        "В «Параметрах» можно уменьшить «Точность ограничения» до 0,000001.",
        "6. «Найти решение» → «Сохранить найденное решение», затем сохранить книгу.",
        "7. Столбец H «Разница» показывает расхождение с Python; ноутбук прочитает веса из столбца B.",
    ]:
        guide.append([line])
    guide.column_dimensions["A"].width = 120
    wb.move_sheet("Инструкция", offset=-2)
    wb.save(path)
    return path


def read_solver_weights(path: Path = SOLVER_PATH) -> pd.Series | None:
    """Веса из листа «Модель» (столбец B). None, если книги нет."""
    if not path.exists():
        return None
    from openpyxl import load_workbook

    ws = load_workbook(path, data_only=False)["Модель"]
    weights = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        name, value = row[0], row[1]
        if name in (None, "Итого"):
            break
        weights[name] = float(value)
    return pd.Series(weights)


def solver_comparison(python_weights: pd.Series, covariance: pd.DataFrame, path: Path = SOLVER_PATH) -> tuple[pd.DataFrame | None, str]:
    """Таблица сравнения весов Python и Excel Solver и текстовый статус."""
    solver = read_solver_weights(path)
    if solver is None:
        return None, "Книга Solver не найдена."
    solver = solver.reindex(python_weights.index)
    n = len(solver)
    if np.allclose(solver.to_numpy(), 1.0 / n, atol=1e-6):
        return None, "Solver ещё не запускался: в книге стоят стартовые веса 1/n."
    cov = covariance.loc[solver.index, solver.index].to_numpy()
    table = pd.DataFrame(
        {
            "Python": python_weights,
            "Excel Solver": solver,
            "Разница": solver - python_weights,
            "Вклад в риск (Solver)": risk_contribution_shares(solver.to_numpy(), cov),
        }
    )
    status = f"Максимальное расхождение весов: {table['Разница'].abs().max():.2e}"
    return table, status
