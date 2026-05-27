"""
Per-item Random Forest sales forecaster.
Features: calendar, weather, holiday, promotion, lag & rolling statistics.
Confidence intervals are derived from the variance across individual trees.
"""
from __future__ import annotations
import os
import json
import math
import warnings
import joblib
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
from datetime import date, timedelta
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy.orm import Session
from database import Sale, WeatherData, Holiday, Promotion, MenuItem

MODEL_DIR = "models"
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
os.makedirs(MODEL_DIR, exist_ok=True)


def _model_path(item_id: int) -> str:
    return os.path.join(MODEL_DIR, f"item_{item_id}.pkl")


# ── Feature engineering ───────────────────────────────────────────────────────

def _build_feature_row(
    dt: date,
    temperature: float,
    precipitation: float,
    is_holiday: bool,
    holiday_factor: float,
    has_promotion: bool,
    promo_discount: float,
    lag_7: float,
    lag_14: float,
    lag_21: float,
    roll_mean_7: float,
    roll_mean_14: float,
    roll_std_7: float,
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


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_sales_df(db: Session) -> pd.DataFrame:
    rows = (
        db.query(Sale.sale_date, Sale.menu_item_id, Sale.quantity_sold)
        .all()
    )
    df = pd.DataFrame(rows, columns=["date", "item_id", "qty"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_context_maps(db: Session):
    """Returns (weather_map, holiday_map, promo_map)."""
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
    """Seasonal temperature estimate when no weather record exists."""
    doy = dt.timetuple().tm_yday
    temp = 10 + 10 * math.sin(2 * math.pi * (doy - 80) / 365)
    return {"temperature": round(temp, 1), "precipitation": 0.0}


# ── Build training dataset for one item ───────────────────────────────────────

def _build_item_dataset(
    item_id: int,
    sales_df: pd.DataFrame,
    weather_map: dict,
    holiday_map: dict,
    promo_map: dict,
) -> tuple[pd.DataFrame, pd.Series]:

    item_df = (
        sales_df[sales_df["item_id"] == item_id]
        .groupby("date")["qty"].sum()
        .rename("qty")
    )

    if item_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    min_date = sales_df["date"].min().date()
    max_date = sales_df["date"].max().date()
    all_dates = pd.date_range(min_date, max_date, freq="D")
    item_series = item_df.reindex(all_dates, fill_value=0)

    rows, targets = [], []
    for i, (idx, qty) in enumerate(item_series.items()):
        dt: date = idx.date()

        if i < 21:  # need at least 21 days of lag
            continue

        past = item_series.iloc[max(0, i - 21): i].values
        lag_7  = float(item_series.iloc[i - 7])  if i >= 7  else float(past.mean())
        lag_14 = float(item_series.iloc[i - 14]) if i >= 14 else float(past.mean())
        lag_21 = float(item_series.iloc[i - 21]) if i >= 21 else float(past.mean())

        last7  = item_series.iloc[max(0, i - 7): i].values
        last14 = item_series.iloc[max(0, i - 14): i].values
        roll_mean_7  = float(last7.mean())  if len(last7)  else lag_7
        roll_mean_14 = float(last14.mean()) if len(last14) else lag_7
        roll_std_7   = float(last7.std())   if len(last7) > 1 else 1.0

        w = weather_map.get(dt, _default_weather(dt))
        h_factor = holiday_map.get(dt, 1.0)
        is_hol = dt in holiday_map
        has_promo, disc = _is_promoted(item_id, dt, promo_map)

        row = _build_feature_row(
            dt, w["temperature"], w["precipitation"],
            is_hol, h_factor, has_promo, disc,
            lag_7, lag_14, lag_21,
            roll_mean_7, roll_mean_14, roll_std_7,
        )
        rows.append(row)
        targets.append(float(qty))

    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    y = pd.Series(targets)
    return X, y


# ── Forecaster class ──────────────────────────────────────────────────────────

class SalesForecaster:
    def __init__(self):
        self.models: dict[int, RandomForestRegressor] = {}
        self.item_ids: list[int] = []
        self.training_metrics: dict[int, dict] = {}
        self.is_trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, db: Session) -> None:
        print("Training sales forecasting models …")
        sales_df = _load_sales_df(db)
        weather_map, holiday_map, promo_map = _load_context_maps(db)

        item_ids = [r.id for r in db.query(MenuItem.id).filter(MenuItem.is_active == True).all()]
        self.item_ids = item_ids

        for item_id in item_ids:
            X, y = _build_item_dataset(item_id, sales_df, weather_map, holiday_map, promo_map)
            if X.empty or len(X) < 30:
                continue

            # Train/test split: last 30 days for evaluation
            split = max(30, int(len(X) * 0.85))
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

            if len(X_test) > 0:
                y_pred = model.predict(X_test)
                mae  = mean_absolute_error(y_test, y_pred)
                rmse = math.sqrt(mean_squared_error(y_test, y_pred))
                # Filter zero-actual rows to avoid divide-by-zero in MAPE
                nonzero = y_test.values > 0
                if nonzero.sum() > 0:
                    mape = float(np.mean(np.abs((y_test.values[nonzero] - y_pred[nonzero]) / y_test.values[nonzero])) * 100)
                else:
                    mape = 0.0
            else:
                mae, rmse, mape = 0.0, 0.0, 0.0

            self.models[item_id] = model
            self.training_metrics[item_id] = {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 1)}
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

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_next_day(
        self, db: Session, target_date: date | None = None
    ) -> list[dict]:
        if target_date is None:
            sales_df = _load_sales_df(db)
            last_date = sales_df["date"].max().date()
            target_date = last_date + timedelta(days=1)

        sales_df = _load_sales_df(db)
        weather_map, holiday_map, promo_map = _load_context_maps(db)

        # Weather for target date: use DB record or seasonal estimate
        w = weather_map.get(target_date, _default_weather(target_date))
        h_factor = holiday_map.get(target_date, 1.0)
        is_hol = target_date in holiday_map

        items = db.query(MenuItem).filter(MenuItem.is_active == True).all()
        results = []

        for item in items:
            item_id = item.id
            if item_id not in self.models:
                continue

            # Compute lag features from historical sales
            item_df = (
                sales_df[sales_df["item_id"] == item_id]
                .groupby("date")["qty"].sum()
            )
            item_dates = pd.date_range(
                sales_df["date"].min(), target_date - timedelta(days=1), freq="D"
            )
            item_series = item_df.reindex(item_dates, fill_value=0)

            n = len(item_series)

            def _lag(k):
                idx = n - k
                return float(item_series.iloc[idx]) if idx >= 0 else float(item_series.mean())

            lag_7, lag_14, lag_21 = _lag(7), _lag(14), _lag(21)
            last7  = item_series.iloc[max(0, n - 7):].values
            last14 = item_series.iloc[max(0, n - 14):].values
            roll_mean_7  = float(last7.mean())  if len(last7) else lag_7
            roll_mean_14 = float(last14.mean()) if len(last14) else lag_7
            roll_std_7   = float(last7.std())   if len(last7) > 1 else 1.0

            has_promo, disc = _is_promoted(item_id, target_date, promo_map)

            feature_row = _build_feature_row(
                target_date, w["temperature"], w["precipitation"],
                is_hol, h_factor, has_promo, disc,
                lag_7, lag_14, lag_21,
                roll_mean_7, roll_mean_14, roll_std_7,
            )
            X = pd.DataFrame([feature_row], columns=FEATURE_COLS)

            model = self.models[item_id]
            X_np = X.values  # avoid sklearn feature-name warning per tree
            # Ensemble predictions from each tree for confidence interval
            tree_preds = np.array([t.predict(X_np)[0] for t in model.estimators_])
            mean_pred  = float(np.mean(tree_preds))
            std_pred   = float(np.std(tree_preds))

            lower = max(0.0, mean_pred - 1.645 * std_pred)   # 90 % CI
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

    # ── Historical evaluation (actual vs predicted) ───────────────────────────

    def historical_accuracy(self, db: Session, last_n_days: int = 30) -> list[dict]:
        sales_df = _load_sales_df(db)
        if sales_df.empty:
            return []

        max_date = sales_df["date"].max().date()
        start_eval = max_date - timedelta(days=last_n_days)

        weather_map, holiday_map, promo_map = _load_context_maps(db)
        items = {i.id: i for i in db.query(MenuItem).filter(MenuItem.is_active == True).all()}

        rows = []
        for item_id, model in self.models.items():
            item = items.get(item_id)
            if not item:
                continue

            item_df = (
                sales_df[sales_df["item_id"] == item_id]
                .groupby("date")["qty"].sum()
            )
            all_dates = pd.date_range(sales_df["date"].min(), max_date, freq="D")
            item_series = item_df.reindex(all_dates, fill_value=0)

            eval_dates = pd.date_range(start_eval, max_date, freq="D")

            for dt_stamp in eval_dates:
                dt = dt_stamp.date()
                i = (dt_stamp - all_dates[0]).days
                if i < 21:
                    continue

                def _lag(k):
                    idx = i - k
                    return float(item_series.iloc[idx]) if idx >= 0 else float(item_series.mean())

                lag_7, lag_14, lag_21 = _lag(7), _lag(14), _lag(21)
                last7  = item_series.iloc[max(0, i - 7): i].values
                last14 = item_series.iloc[max(0, i - 14): i].values
                roll_mean_7  = float(last7.mean())  if len(last7) else lag_7
                roll_mean_14 = float(last14.mean()) if len(last14) else lag_7
                roll_std_7   = float(last7.std())   if len(last7) > 1 else 1.0

                w = weather_map.get(dt, _default_weather(dt))
                h_factor = holiday_map.get(dt, 1.0)
                is_hol = dt in holiday_map
                has_promo, disc = _is_promoted(item_id, dt, promo_map)

                feat = _build_feature_row(
                    dt, w["temperature"], w["precipitation"],
                    is_hol, h_factor, has_promo, disc,
                    lag_7, lag_14, lag_21,
                    roll_mean_7, roll_mean_14, roll_std_7,
                )
                X = pd.DataFrame([feat], columns=FEATURE_COLS)
                pred = float(model.predict(X)[0])
                actual = float(item_series.iloc[i])

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
        items = {i.id: i.name for i in db.query(MenuItem).all()}
        out = []
        for item_id, model in self.models.items():
            imp = dict(zip(FEATURE_COLS, model.feature_importances_))
            top = sorted(imp.items(), key=lambda x: -x[1])[:5]
            out.append({"item_id": item_id, "item_name": items.get(item_id, "?"), "top_features": top})
        return out
