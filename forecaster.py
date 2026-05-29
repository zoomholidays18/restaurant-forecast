"""
Per-item sales forecaster using pure numpy (bootstrap ridge regression).
No scikit-learn dependency — fast to install and deploy.
"""
from __future__ import annotations
import os
import json
import math
import numpy as np
from datetime import date, timedelta
from collections import defaultdict
from sqlalchemy.orm import Session
from database import Sale, WeatherData, Holiday, Promotion, MenuItem
import joblib

MODEL_DIR = "models"
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
os.makedirs(MODEL_DIR, exist_ok=True)

N_BOOTSTRAP = 100   # bootstrap samples for confidence intervals
ALPHA = 10.0        # ridge regularisation strength


def _model_path(item_id: int) -> str:
    return os.path.join(MODEL_DIR, f"item_{item_id}.pkl")


def _date_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


# ── Feature engineering ───────────────────────────────────────────────────────

def _build_feature_row(
    dt: date, temperature: float, precipitation: float,
    is_holiday: bool, holiday_factor: float,
    has_promotion: bool, promo_discount: float,
    lag_7: float, lag_14: float, lag_21: float,
    roll_mean_7: float, roll_mean_14: float, roll_std_7: float,
) -> dict:
    dow = dt.weekday()
    return {
        "day_of_week":    dow,
        "is_weekend":     int(dow >= 5),
        "month":          dt.month,
        "day_of_month":   dt.day,
        "week_of_year":   dt.isocalendar()[1],
        "sin_dow":        math.sin(2 * math.pi * dow / 7),
        "cos_dow":        math.cos(2 * math.pi * dow / 7),
        "sin_month":      math.sin(2 * math.pi * dt.month / 12),
        "cos_month":      math.cos(2 * math.pi * dt.month / 12),
        "temperature":    temperature,
        "precipitation":  precipitation,
        "is_raining":     int(precipitation > 1),
        "is_holiday":     int(is_holiday),
        "holiday_factor": holiday_factor,
        "has_promotion":  int(has_promotion),
        "promo_discount": promo_discount,
        "lag_7":          lag_7,
        "lag_14":         lag_14,
        "lag_21":         lag_21,
        "roll_mean_7":    roll_mean_7,
        "roll_mean_14":   roll_mean_14,
        "roll_std_7":     roll_std_7,
    }


FEATURE_COLS = list(_build_feature_row(
    date.today(), 20, 0, False, 1.0, False, 0.0, 10, 10, 10, 10, 10, 1.0
).keys())


# ── Ridge regression (pure numpy) ─────────────────────────────────────────────

class RidgeModel:
    def __init__(self, alpha: float = ALPHA):
        self.alpha = alpha
        self.params: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeModel":
        n, p = X.shape
        Xb = np.column_stack([np.ones(n), X])
        A = Xb.T @ Xb + self.alpha * np.eye(p + 1)
        A[0, 0] -= self.alpha          # don't regularise bias
        self.params = np.linalg.solve(A, Xb.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xb = np.column_stack([np.ones(len(X)), X])
        return np.maximum(0.0, Xb @ self.params)


class BootstrapForecaster:
    """Ensemble of ridge models on bootstrap samples for uncertainty."""

    def __init__(self, n_estimators: int = N_BOOTSTRAP):
        self.n_estimators = n_estimators
        self.estimators: list[RidgeModel] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BootstrapForecaster":
        rng = np.random.default_rng(42)
        n = len(X)
        self.estimators = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            m = RidgeModel().fit(X[idx], y[idx])
            self.estimators.append(m)
        return self

    def predict_with_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = np.stack([m.predict(X) for m in self.estimators])
        return preds.mean(axis=0), preds.std(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_with_std(X)[0]


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


def _item_series(item_id: int, sales_dict: dict, date_list: list[date]) -> np.ndarray:
    item_data = sales_dict.get(item_id, {})
    return np.array([float(item_data.get(d, 0.0)) for d in date_list], dtype=float)


def _build_item_dataset(
    item_id: int, sales_dict: dict, date_list: list[date],
    weather_map: dict, holiday_map: dict, promo_map: dict,
) -> tuple[np.ndarray, np.ndarray]:
    series = _item_series(item_id, sales_dict, date_list)
    if np.sum(series) == 0:
        return np.empty((0, len(FEATURE_COLS))), np.array([])

    X_rows, y_vals = [], []
    for i, dt in enumerate(date_list):
        if i < 21:
            continue
        lag_7, lag_14, lag_21 = series[i-7], series[i-14], series[i-21]
        last7  = series[max(0, i-7):i]
        last14 = series[max(0, i-14):i]
        roll_mean_7  = float(np.mean(last7))  if len(last7)  else float(lag_7)
        roll_mean_14 = float(np.mean(last14)) if len(last14) else float(lag_7)
        roll_std_7   = float(np.std(last7))   if len(last7) > 1 else 1.0
        w = weather_map.get(dt, _default_weather(dt))
        h_factor = holiday_map.get(dt, 1.0)
        row = _build_feature_row(
            dt, w["temperature"], w["precipitation"],
            dt in holiday_map, h_factor,
            *_is_promoted(item_id, dt, promo_map),
            float(lag_7), float(lag_14), float(lag_21),
            roll_mean_7, roll_mean_14, roll_std_7,
        )
        X_rows.append(list(row.values()))
        y_vals.append(series[i])

    if not X_rows:
        return np.empty((0, len(FEATURE_COLS))), np.array([])
    return np.array(X_rows, dtype=float), np.array(y_vals, dtype=float)


# ── Forecaster class ──────────────────────────────────────────────────────────

class SalesForecaster:
    def __init__(self):
        self.models: dict[int, BootstrapForecaster] = {}
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

            model = BootstrapForecaster().fit(X_tr, y_tr)

            if len(X_te) > 0:
                y_pred = model.predict(X_te)
                mae  = float(np.mean(np.abs(y_te - y_pred)))
                rmse = float(np.sqrt(np.mean((y_te - y_pred) ** 2)))
                nz = y_te > 0
                mape = float(np.mean(np.abs((y_te[nz] - y_pred[nz]) / y_te[nz])) * 100) if nz.sum() > 0 else 0.0
            else:
                mae = rmse = mape = 0.0

            self.models[item_id] = model
            self.training_metrics[item_id] = {
                "mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 1)
            }
            joblib.dump(model, _model_path(item_id))

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
                self.models[item_id] = joblib.load(p)
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
            series = _item_series(item_id, sales_dict, date_list) if date_list else np.array([])
            n = len(series)

            def _lag(k):
                idx = n - k
                return float(series[idx]) if idx >= 0 and n > 0 else (float(np.mean(series)) if n > 0 else 0.0)

            lag_7, lag_14, lag_21 = _lag(7), _lag(14), _lag(21)
            last7  = series[max(0, n-7):]  if n > 0 else np.array([])
            last14 = series[max(0, n-14):] if n > 0 else np.array([])
            roll_mean_7  = float(np.mean(last7))  if len(last7)  else lag_7
            roll_mean_14 = float(np.mean(last14)) if len(last14) else lag_7
            roll_std_7   = float(np.std(last7))   if len(last7) > 1 else 1.0

            has_promo, disc = _is_promoted(item_id, target_date, promo_map)
            feat = _build_feature_row(
                target_date, w["temperature"], w["precipitation"],
                is_hol, h_factor, has_promo, disc,
                lag_7, lag_14, lag_21, roll_mean_7, roll_mean_14, roll_std_7,
            )
            X_np = np.array([list(feat.values())], dtype=float)

            mean_pred, std_pred = self.models[item_id].predict_with_std(X_np)
            mean_pred, std_pred = float(mean_pred[0]), float(std_pred[0])
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
                lag_7, lag_14, lag_21 = float(series[i-7]), float(series[i-14]), float(series[i-21])
                last7  = series[max(0, i-7):i]
                last14 = series[max(0, i-14):i]
                roll_mean_7  = float(np.mean(last7))  if len(last7)  else lag_7
                roll_mean_14 = float(np.mean(last14)) if len(last14) else lag_7
                roll_std_7   = float(np.std(last7))   if len(last7) > 1 else 1.0
                w = weather_map.get(dt, _default_weather(dt))
                h_factor = holiday_map.get(dt, 1.0)
                feat = _build_feature_row(
                    dt, w["temperature"], w["precipitation"],
                    dt in holiday_map, h_factor,
                    *_is_promoted(item_id, dt, promo_map),
                    lag_7, lag_14, lag_21, roll_mean_7, roll_mean_14, roll_std_7,
                )
                X_np = np.array([list(feat.values())], dtype=float)
                pred   = float(model.predict(X_np)[0])
                actual = float(series[i])
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
