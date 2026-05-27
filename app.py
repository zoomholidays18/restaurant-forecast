"""
ForkCast AI – Restaurant Sales Prediction & Inventory Management
Run: python app.py
"""
import os
import threading
import webbrowser
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Optional

import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import (
    init_db, get_db, SessionLocal,
    MenuItem, Ingredient, Sale, WeatherData, Holiday, Promotion, Prediction,
    BillOfMaterials,
)
from seed_data import seed_database
from forecaster import SalesForecaster
from inventory_engine import InventoryEngine
from weather_service import refresh_weather, load_config, save_config, fetch_forecast

# ── Globals ───────────────────────────────────────────────────────────────────
forecaster = SalesForecaster()
engine_inv = InventoryEngine()
_training_lock = threading.Lock()
_is_training = False


def _startup_train():
    global _is_training
    with _training_lock:
        _is_training = True
    db = SessionLocal()
    try:
        if not forecaster.load_saved(db):
            forecaster.train(db)
    finally:
        db.close()
        with _training_lock:
            _is_training = False
    print("Ready.")


def _startup_weather():
    """Fetch real weather forecast in background at startup."""
    db = SessionLocal()
    try:
        result = refresh_weather(db)
        print(f"Weather: fetched {result['stored']} days from Open-Meteo "
              f"({result['location']['lat']}, {result['location']['lon']})")
    except Exception as exc:
        print(f"Weather fetch skipped (offline?): {exc}")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    seed_database()
    threading.Thread(target=_startup_weather, daemon=True).start()
    threading.Thread(target=_startup_train,   daemon=True).start()
    yield


app = FastAPI(title="ForkCast AI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


# ── Status ─────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    return {"trained": forecaster.is_trained, "training": _is_training}


# ── Menu items ─────────────────────────────────────────────────────────────────

@app.get("/api/menu-items")
def get_menu_items(db: Session = Depends(get_db)):
    items = db.query(MenuItem).filter(MenuItem.is_active == True).all()
    return [
        {"id": i.id, "name": i.name, "category": i.category,
         "base_price": i.base_price, "description": i.description}
        for i in items
    ]


@app.get("/api/menu-items/{item_id}/bom")
def get_bom(item_id: int, db: Session = Depends(get_db)):
    rows = (
        db.query(BillOfMaterials, Ingredient)
        .join(Ingredient, BillOfMaterials.ingredient_id == Ingredient.id)
        .filter(BillOfMaterials.menu_item_id == item_id)
        .all()
    )
    return [
        {
            "ingredient_id": ing.id,
            "ingredient_name": ing.name,
            "unit": ing.unit,
            "quantity": bom.quantity,
            "cost_per_unit": ing.cost_per_unit,
        }
        for bom, ing in rows
    ]


# ── Ingredients & inventory ────────────────────────────────────────────────────

@app.get("/api/ingredients")
def get_ingredients(db: Session = Depends(get_db)):
    ings = db.query(Ingredient).all()
    return [
        {
            "id": i.id, "name": i.name, "unit": i.unit,
            "cost_per_unit": i.cost_per_unit, "shelf_life_days": i.shelf_life_days,
            "min_order_qty": i.min_order_qty, "current_stock": i.current_stock,
        }
        for i in ings
    ]


class StockUpdate(BaseModel):
    quantity: float


@app.put("/api/ingredients/{ingredient_id}/stock")
def update_stock(ingredient_id: int, body: StockUpdate, db: Session = Depends(get_db)):
    ing = engine_inv.update_stock(db, ingredient_id, body.quantity)
    if not ing:
        raise HTTPException(404, "Ingredient not found")
    return {"id": ing.id, "name": ing.name, "current_stock": ing.current_stock}


# ── Predictions ────────────────────────────────────────────────────────────────

@app.get("/api/predictions/next-day")
def predict_next_day(target_date: Optional[str] = None, db: Session = Depends(get_db)):
    if not forecaster.is_trained:
        return JSONResponse({"detail": "Models still training, please wait."}, status_code=202)

    dt = None
    if target_date:
        try:
            dt = date.fromisoformat(target_date)
        except ValueError:
            raise HTTPException(400, "Invalid date format, use YYYY-MM-DD")

    predictions = forecaster.predict_next_day(db, dt)
    return predictions


class PredictionOverride(BaseModel):
    item_id: int
    predicted_qty: float
    target_date: Optional[str] = None


@app.post("/api/predictions/override")
def override_prediction(body: PredictionOverride, db: Session = Depends(get_db)):
    target_date = date.fromisoformat(body.target_date) if body.target_date else (date.today() + timedelta(days=1))
    pred = (
        db.query(Prediction)
        .filter(Prediction.menu_item_id == body.item_id,
                Prediction.prediction_date == target_date)
        .first()
    )
    if pred:
        pred.predicted_qty = body.predicted_qty
        pred.is_manual_override = True
    else:
        pred = Prediction(
            prediction_date=target_date,
            menu_item_id=body.item_id,
            predicted_qty=body.predicted_qty,
            is_manual_override=True,
        )
        db.add(pred)
    db.commit()
    return {"status": "ok", "item_id": body.item_id, "qty": body.predicted_qty}


# ── Inventory recommendations ──────────────────────────────────────────────────

@app.get("/api/inventory/recommendations")
def inventory_recommendations(target_date: Optional[str] = None, db: Session = Depends(get_db)):
    if not forecaster.is_trained:
        return JSONResponse({"detail": "Models still training, please wait."}, status_code=202)

    dt = None
    if target_date:
        try:
            dt = date.fromisoformat(target_date)
        except ValueError:
            raise HTTPException(400, "Invalid date format")

    predictions = forecaster.predict_next_day(db, dt)
    recommendations = engine_inv.calculate_requirements(predictions, db)
    summary = engine_inv.summary_metrics(predictions, recommendations)

    return {"summary": summary, "recommendations": recommendations, "predictions": predictions}


# ── Historical accuracy ────────────────────────────────────────────────────────

@app.get("/api/model/performance")
def model_performance(last_n_days: int = 30, db: Session = Depends(get_db)):
    if not forecaster.is_trained:
        return JSONResponse({"detail": "Models still training."}, status_code=202)
    rows = forecaster.historical_accuracy(db, last_n_days)
    # Aggregate per item
    from collections import defaultdict
    agg = defaultdict(lambda: {"errors": [], "name": ""})
    for r in rows:
        agg[r["item_id"]]["errors"].append(r["error"])
        agg[r["item_id"]]["name"] = r["item_name"]

    metrics = []
    for item_id, d in agg.items():
        errs = d["errors"]
        import numpy as np
        metrics.append({
            "item_id":   item_id,
            "item_name": d["name"],
            "mae":       round(float(np.mean(errs)), 2),
            "rmse":      round(float(np.sqrt(np.mean([e**2 for e in errs]))), 2),
            "n_days":    len(errs),
        })

    return {"detail_rows": rows, "metrics": metrics}


@app.get("/api/model/training-metrics")
def training_metrics():
    return forecaster.training_metrics


# ── Sales history ──────────────────────────────────────────────────────────────

@app.get("/api/sales/history")
def sales_history(days: int = 30, db: Session = Depends(get_db)):
    from sqlalchemy import func
    cutoff = date.today() - timedelta(days=days)
    rows = (
        db.query(
            Sale.sale_date,
            MenuItem.name,
            MenuItem.category,
            func.sum(Sale.quantity_sold).label("total_qty"),
            func.sum(Sale.quantity_sold * Sale.unit_price).label("revenue"),
        )
        .join(MenuItem, Sale.menu_item_id == MenuItem.id)
        .filter(Sale.sale_date >= cutoff)
        .group_by(Sale.sale_date, MenuItem.name, MenuItem.category)
        .order_by(Sale.sale_date.desc())
        .all()
    )
    return [
        {
            "date":     r.sale_date.isoformat(),
            "item":     r.name,
            "category": r.category,
            "qty":      int(r.total_qty),
            "revenue":  round(float(r.revenue), 2),
        }
        for r in rows
    ]


@app.get("/api/sales/daily-totals")
def daily_totals(days: int = 90, db: Session = Depends(get_db)):
    from sqlalchemy import func
    cutoff = date.today() - timedelta(days=days)
    rows = (
        db.query(
            Sale.sale_date,
            func.sum(Sale.quantity_sold).label("total_qty"),
            func.sum(Sale.quantity_sold * Sale.unit_price).label("revenue"),
        )
        .filter(Sale.sale_date >= cutoff)
        .group_by(Sale.sale_date)
        .order_by(Sale.sale_date)
        .all()
    )
    return [
        {"date": r.sale_date.isoformat(), "qty": int(r.total_qty), "revenue": round(float(r.revenue), 2)}
        for r in rows
    ]


# ── Model retraining ───────────────────────────────────────────────────────────

@app.post("/api/model/retrain")
def retrain(db: Session = Depends(get_db)):
    if _is_training:
        return {"status": "already_training"}

    def _bg():
        global _is_training
        with _training_lock:
            _is_training = True
        db2 = SessionLocal()
        try:
            forecaster.train(db2)
        finally:
            db2.close()
            with _training_lock:
                _is_training = False

    t = threading.Thread(target=_bg, daemon=True)
    t.start()
    return {"status": "retraining_started"}


# ── Promotions & holidays ──────────────────────────────────────────────────────

@app.get("/api/promotions")
def get_promotions(db: Session = Depends(get_db)):
    promos = db.query(Promotion).all()
    return [
        {
            "id": p.id, "name": p.name,
            "menu_item_id": p.menu_item_id,
            "start_date": p.start_date.isoformat(),
            "end_date": p.end_date.isoformat(),
            "discount_pct": p.discount_pct,
        }
        for p in promos
    ]


@app.get("/api/holidays")
def get_holidays(db: Session = Depends(get_db)):
    hols = db.query(Holiday).all()
    return [
        {"date": h.holiday_date.isoformat(), "name": h.name, "impact_factor": h.impact_factor}
        for h in hols
    ]


# ── Weather ───────────────────────────────────────────────────────────────────

@app.get("/api/weather/forecast")
def get_weather_forecast(days: int = 7, db: Session = Depends(get_db)):
    """Return upcoming weather from the DB (populated by Open-Meteo)."""
    from sqlalchemy import asc
    today = date.today()
    rows = (
        db.query(WeatherData)
        .filter(WeatherData.weather_date >= today)
        .order_by(asc(WeatherData.weather_date))
        .limit(days)
        .all()
    )
    from weather_service import CONDITION_EMOJI
    return [
        {
            "date":          w.weather_date.isoformat(),
            "temperature":   w.temperature,
            "precipitation": w.precipitation,
            "condition":     w.condition,
            "emoji":         CONDITION_EMOJI.get(w.condition, "🌤️"),
        }
        for w in rows
    ]


@app.post("/api/weather/refresh")
def weather_refresh(db: Session = Depends(get_db)):
    """Pull fresh forecast from Open-Meteo and store it."""
    try:
        result = refresh_weather(db)
        return result
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))


class WeatherUpdate(BaseModel):
    temperature: float
    precipitation: float
    condition: str = "sunny"


@app.post("/api/weather/manual")
def set_manual_weather(body: WeatherUpdate, target_date: Optional[str] = None,
                       db: Session = Depends(get_db)):
    """Override a specific day's weather manually."""
    dt = date.fromisoformat(target_date) if target_date else date.today() + timedelta(days=1)
    existing = db.query(WeatherData).filter(WeatherData.weather_date == dt).first()
    if existing:
        existing.temperature  = body.temperature
        existing.precipitation = body.precipitation
        existing.condition     = body.condition
    else:
        db.add(WeatherData(weather_date=dt, temperature=body.temperature,
                           precipitation=body.precipitation, condition=body.condition))
    db.commit()
    return {"status": "ok", "date": dt.isoformat()}


# ── Restaurant config ─────────────────────────────────────────────────────────

@app.get("/api/config")
def get_config():
    return load_config()


class ConfigUpdate(BaseModel):
    restaurant_name: Optional[str] = None
    latitude:        Optional[float] = None
    longitude:       Optional[float] = None
    timezone:        Optional[str] = None
    currency:        Optional[str] = None


@app.post("/api/config")
def update_config(body: ConfigUpdate, db: Session = Depends(get_db)):
    cfg = load_config()
    if body.restaurant_name is not None: cfg["restaurant_name"] = body.restaurant_name
    if body.latitude        is not None: cfg["latitude"]        = body.latitude
    if body.longitude       is not None: cfg["longitude"]       = body.longitude
    if body.timezone        is not None: cfg["timezone"]        = body.timezone
    if body.currency        is not None: cfg["currency"]        = body.currency
    save_config(cfg)

    # Immediately fetch weather for new location
    try:
        result = refresh_weather(db)
        return {"status": "ok", "config": cfg, "weather_updated": result["stored"]}
    except RuntimeError as exc:
        return {"status": "ok", "config": cfg, "weather_warning": str(exc)}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8000))
    IS_LOCAL = "PORT" not in os.environ
    print(f"Starting ForkCast AI on http://localhost:{PORT}")
    if IS_LOCAL:
        threading.Timer(3.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
