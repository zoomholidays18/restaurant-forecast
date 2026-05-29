"""
Per-item sales forecaster — pure Python, no numpy, no scikit-learn.
Uses ridge regression with Gaussian elimination + residual-based confidence intervals.
"""
from __future__ import annotations
import os
import json
import math
import random
import pickle
from datetime import date, timedelta
from collections import defaultdict
from sqlalchemy.orm import Session
from database import Sale, WeatherData, Holiday, Promotion, MenuItem

MODEL_DIR = "models"
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
os.makedirs(MODEL_DIR, exist_ok=True)

ALPHA = 10.0   # ridge regularisation


def _model_path(item_id: int) -> str:
    return os.path.join(MODEL_DIR, f"item_{item_id}.pkl")


def _date_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


# ── Pure-Python linear algebra ────────────────────────────────────────────────

def _dot(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


def _mat_xtx(X: list[list]) -> list[list]:
    p = len(X[0])
    result = [[0.0] * p for _ in range(p)]
    for row in X:
        for i in range(p):
            for j in range(p):
                result[i][j] += row[i] * row[j]
    return result


def _mat_xty(X: list[list], y: list) -> list:
    p = len(X[0])
    result = [0.0] * p
    for row, yi in zip(X, y):
        for i in range(p):
            result[i] += row[i] * yi
    return result


def _solve(A: list[list], b: list) -> list:
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [[A[i][j] for j in range(n)] + [float(b[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        if abs(M[col][col]) < 1e-12:
            continue
        for row in range(col + 1, n):
            f = M[row][col] / M[col][col]
            for j in range(col, n + 1):
                M[row][j] -= f * M[col][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        if abs(M[i][i]) > 1e-12:
            x[i] /= M[i][i]
    return x


# ── Ridge regression model ────────────────────────────────────────────────────

class RidgeModel:
    def __init__(self, alpha: float = ALPHA):
        self.alpha = alpha
        self.params: list[float] = []
        self.residual_std: float = 1.0

    def fit(self, X: list[list], y: list) -> "RidgeModel":
        n, p = len(X), len(X[0])
        Xb = [[1.0] + row for row in X]
        pb = p + 1
        XtX = _mat_xtx(Xb)
        for i in range(1, pb):
            XtX[i][i] += self.alpha
        Xty = _mat_xty(Xb, y)
        self.params = _solve(XtX, Xty)
        preds = [max(0.0, _dot(self.params, row)) for row in Xb]
        residuals = [yi - pi for yi, pi in zip(y, preds)]
        mean_r = sum(residuals) / len(residuals) if residuals else 0.0
        var_r = sum((r - mean_r) ** 2 for r in residuals) / len(residuals) if residuals else 1.0
        self.residual_std = max(0.5, math.sqrt(var_r))
        return self

    def predict(self, x: list) -> float:
        return max(0.0, _dot(self.params, [1.0] + x))


# ── Feature engineering ───────────────────────────────────────────────────────

def _build_feature_row(
    dt: date, temperature: float, precipitation: float,
    is_holiday: bool, holiday_factor: float,
    has_promotion: bool, promo_discount: float,
    lag_7: float, lag_14: float, lag_21: float,
    roll_mean_7: float, roll_mean_14: float, roll_std_7: float,
) -> list:
    dow = dt.weekday()
    return [
        dow,
        int(dow >= 5),
        dt.month,
        dt.day,
        dt.isocalendar()[1],
        math.sin(2 * math.pi * dow / 7),
        math.cos(2 * math.pi * dow / 7),
        math.sin(2 * math.pi * dt.month / 12),
        math.cos(2 * math.pi * dt.month / 12),
        temperature,
        precipitation,
        int(precipitation > 1),
        int(is_holiday),
        holiday_factor,
        int(has_promotion),
        promo_discount,
        lag_7, lag_14, lag_21,
        roll_mean_7, roll_mean_14, roll_std_7,
    ]


FEATURE_COLS = [
    "day_of_week", "is_weekend", "month", "day_of_month", "week_of_year",
    "sin_dow", "cos_dow", "sin_month", "cos_month",
    "temperature", "precipitation", "is_raining",
    "is_holiday", "holiday_factor", "has_promotion", "promo_discount",
    "lag_7", "lag_14", "lag_21", "roll_mean_7", "roll_mean_14", "roll_std_7",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_sales_data(db: Session):
    rows = db.query(Sale.sale_date, Sale.menu_item_id, Sale.quantity_sold).all()
    sales_dict: dict[int, dict[date, float]] = defaultdict(lambda: defaultdict(float))
    all_dates: list[date] = []
    for sale_date, item_id, qty in rows:
        d = sale_date if isinstance(sale_date, date) else sale_date
        sales_dict[item_id][d] += float(qty)
        all_dates.append(d)
    if not all_dates:
        return sales_dict, None, None
    return sales_dict, min(all_dates), max(all_dates)


def _load_context_maps(db: Session):
    weather_map: dict[date, dict] = {}
    for w in db.query(WeatherData).all():
        weather_map[w.weather_date] = {
            "temperature": w.temperature or 18.0,
            "precipitation": w.precipitation or 0.0,
        }
    holiday_map: dict[date, float] = {}
    for h in db.query(Holiday).all():
        holiday_map[h.holiday_date] = h.impact_factor
    promo_map: dict[int, list] = defaultdict(list)
    for p in db.query(Promotion).all():
        promo_map[p.menu_item_id].append(
            (p.start_date, p.end_date, p.discount_pct, p.name)
        )
    return weather_map, holiday_map, promo_map


def _is_promoted(item_id: int, dt: date, promo_map: dict) -> tuple[bool, float]:
    dow = dt.weekday()
    for (ps, pe, pct, pname) in promo_map.get(item_id, []):
        if ps <= dt <= pe:
            if "Tuesday" in pname and dow != 1:
                continue
            if "Weekend" in pname and dow not in (4, 5, 6):
                continue
            return True, pct
    return False, 0.0


def _default_weather(dt: date) -> dict:
    doy = dt.timetuple().tm_yday
    temp = 10 + 10 * math.sin(2 * math.pi * (doy - 80) / 365)
    return {"temperature": round(temp, 1), "precipitation": 0.0}


def _item_series(item_id: int, sales_dict: dict, date_list: list[date]) -> list[float]:
    item_data = sales_dict.get(item_id, {})
    return [float(item_data.get(d, 0.0)) for d in date_list]


def _list_mean(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _list_std(vals: list) -> float:
    if len(vals) < 2:
        return 1.0
    m = _list_mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def _build_item_dataset(
    item_id: int, sales_dict: dict, date_list: list[date],
    weather_map: dict, holiday_map: dict, promo_map: dict,
) -> tuple[list[list], list]:
    series = _item_series(item_id, sales_dict, date_list)
    if sum(series) == 0:
        return [], []

    X_rows, y_vals = [], []
    for i, dt in enumerate(date_list):
        if i < 21:
            continue
        lag_7, lag_14, lag_21 = series[i-7], series[i-14], series[i-21]
        last7  = series[max(0, i-7):i]
        last14 = series[max(0, i-14):i]
        roll_mean_7  = _list_mean(last7)  if last7  else lag_7
        roll_mean_14 = _list_mean(last14) if last14 else lag_7
        roll_std_7   = _list_std(last7)
        w = weather_map.get(dt, _default_weather(dt))
        has_promo, disc = _is_promoted(item_id, dt, promo_map)
        row = _build_feature_row(
            dt, w["temperature"], w["precipitation"],
            dt in holiday_map, holiday_map.get(dt, 1.0),
            has_promo, disc,
            lag_7, lag_14, lag_21,
            roll_mean_7, roll_mean_14, roll_std_7,
        )
        X_rows.append(row)
        y_vals.append(series[i])

    return X_rows, y_vals


# ── Forecaster class ──────────────────────────────────────────────────────────

class SalesForecaster:
    def __init__(self):
        self.models: dict[int, RidgeModel] = {}
        self.item_ids: list[int] = []
        self.training_metrics: dict[int, dict] = {}
        self.is_trained = False

    def train(self, db: Session) -> None:
        print("Training sales forecasting models …")
        sales_dict, min_date, max_date = _load_sales_data(db)
        if min_date is None:
            print("No sales data.")
            return
        date_list = _date_range(min_date, max_date)
        weather_map, holiday_map, promo_map = _load_context_maps(db)
        item_ids = [r.id for r in db.query(MenuItem.id).filter(MenuItem.is_active == True).all()]
        self.item_ids = item_ids

        for item_id in item_ids:
            X, y = _build_item_dataset(item_id, sales_dict, date_list, weather_map, holiday_map, promo_map)
            if len(X) < 30:
                continue
            split = max(30, int(len(X) * 0.85))
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]

            model = RidgeModel().fit(X_tr, y_tr)

            if X_te:
                y_pred = [model.predict(x) for x in X_te]
                errors = [abs(a - p) for a, p in zip(y_te, y_pred)]
                sq_errors = [(a - p) ** 2 for a, p in zip(y_te, y_pred)]
                mae  = sum(errors) / len(errors)
                rmse = math.sqrt(sum(sq_errors) / len(sq_errors))
                nz = [(a, p) for a, p in zip(y_te, y_pred) if a > 0]
                mape = sum(abs(a - p) / a for a, p in nz) / len(nz) * 100 if nz else 0.0
            else:
                mae = rmse = mape = 0.0

            self.models[item_id] = model
            self.training_metrics[item_id] = {
                "mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 1)
            }
            with open(_model_path(item_id), "wb") as f:
                pickle.dump(model, f)

        self.is_trained = True
        with open(METRICS_PATH, "w") as fh:
            json.dump({str(k): v for k, v in self.training_metrics.items()}, fh)
        print(f"Trained {len(self.models)} item models.")

    def load_saved(self, db: Session) -> bool:
        item_ids = [r.id for r in db.query(MenuItem.id).filter(MenuItem.is_active == True).all()]
        self.item_ids = item_ids
        loaded = 0
        for item_id in item_ids:
            p = _model_path(item_id)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    self.models[item_id] = pickle.load(f)
                loaded += 1
        if loaded == len(item_ids) and loaded > 0:
            self.is_trained = True
            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH) as fh:
                    raw = json.load(fh)
                self.training_metrics = {int(k): v for k, v in raw.items()}
            print(f"Loaded {loaded} saved models.")
            return True
        return False

    def predict_next_day(self, db: Session, target_date: date | None = None) -> list[dict]:
        sales_dict, min_date, max_date = _load_sales_data(db)
        if target_date is None:
            target_date = (max_date + timedelta(days=1)) if max_date else (date.today() + timedelta(days=1))
        date_list = _date_range(min_date, target_date - timedelta(days=1)) if min_date else []
        weather_map, holiday_map, promo_map = _load_context_maps(db)

        w = weather_map.get(target_date, _default_weather(target_date))
        h_factor = holiday_map.get(target_date, 1.0)
        is_hol = target_date in holiday_map
        items = db.query(MenuItem).filter(MenuItem.is_active == True).all()
        results = []

        for item in items:
            item_id = item.id
            if item_id not in self.models:
                continue
            series = _item_series(item_id, sales_dict, date_list) if date_list else []
            n = len(series)

            def _lag(k):
                idx = n - k
                return float(series[idx]) if 0 <= idx < n else (_list_mean(series) if series else 0.0)

            lag_7, lag_14, lag_21 = _lag(7), _lag(14), _lag(21)
            last7  = series[max(0, n-7):]  if n > 0 else []
            last14 = series[max(0, n-14):] if n > 0 else []
            roll_mean_7  = _list_mean(last7)  if last7  else lag_7
            roll_mean_14 = _list_mean(last14) if last14 else lag_7
            roll_std_7   = _list_std(last7)

            has_promo, disc = _is_promoted(item_id, target_date, promo_map)
            feat = _build_feature_row(
                target_date, w["temperature"], w["precipitation"],
                is_hol, h_factor, has_promo, disc,
                lag_7, lag_14, lag_21, roll_mean_7, roll_mean_14, roll_std_7,
            )
            model = self.models[item_id]
            mean_pred = model.predict(feat)
            std_pred  = model.residual_std

            lower = max(0.0, mean_pred - 1.645 * std_pred)
            upper = mean_pred + 1.645 * std_pred
            metrics = self.training_metrics.get(item_id, {})

            results.append({
                "item_id":        item_id,
                "item_name":      item.name,
                "category":       item.category,
                "base_price":     item.base_price,
                "predicted_qty":  round(mean_pred, 1),
                "lower_bound":    round(lower, 1),
                "upper_bound":    round(upper, 1),
                "std":            round(std_pred, 2),
                "has_promotion":  has_promo,
                "promo_discount": disc,
                "model_mae":      metrics.get("mae", 0),
                "model_rmse":     metrics.get("rmse", 0),
                "model_mape":     metrics.get("mape", 0),
            })
        return results

    def historical_accuracy(self, db: Session, last_n_days: int = 30) -> list[dict]:
        sales_dict, min_date, max_date = _load_sales_data(db)
        if min_date is None:
            return []
        start_eval = max_date - timedelta(days=last_n_days)
        date_list = _date_range(min_date, max_date)
        weather_map, holiday_map, promo_map = _load_context_maps(db)
        items = {i.id: i for i in db.query(MenuItem).filter(MenuItem.is_active == True).all()}

        rows = []
        for item_id, model in self.models.items():
            item = items.get(item_id)
            if not item:
                continue
            series = _item_series(item_id, sales_dict, date_list)
            for i, dt in enumerate(date_list):
                if dt < start_eval or i < 21:
                    continue
                lag_7, lag_14, lag_21 = series[i-7], series[i-14], series[i-21]
                last7  = series[max(0, i-7):i]
                last14 = series[max(0, i-14):i]
                roll_mean_7  = _list_mean(last7)  if last7  else lag_7
                roll_mean_14 = _list_mean(last14) if last14 else lag_7
                roll_std_7   = _list_std(last7)
                w = weather_map.get(dt, _default_weather(dt))
                feat = _build_feature_row(
                    dt, w["temperature"], w["precipitation"],
                    dt in holiday_map, holiday_map.get(dt, 1.0),
                    *_is_promoted(item_id, dt, promo_map),
                    lag_7, lag_14, lag_21, roll_mean_7, roll_mean_14, roll_std_7,
                )
                pred   = model.predict(feat)
                actual = series[i]
                rows.append({
                    "date":      dt.isoformat(),
                    "item_id":   item_id,
                    "item_name": item.name,
                    "predicted": round(pred, 1),
                    "actual":    int(actual),
                    "error":     round(abs(pred - actual), 1),
                    "error_pct": round(abs(pred - actual) / max(actual, 1) * 100, 1),
                })
        return rows

    def get_feature_importances(self, db: Session) -> list[dict]:
        return []
