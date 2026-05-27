"""
Open-Meteo weather integration (free, no API key required).
Docs: https://open-meteo.com/en/docs

Fetches 7-day daily forecasts and upserts into the WeatherData table.
"""
from __future__ import annotations
import json
import urllib.request
import urllib.parse
from datetime import date
from sqlalchemy.orm import Session
from database import WeatherData

CONFIG_PATH = "config.json"

DEFAULT_CONFIG = {
    "restaurant_name": "The Fork Restaurant",
    "latitude": 51.5074,
    "longitude": -0.1278,
    "timezone": "Europe/London",
    "currency": "USD",
}

# WMO Weather Interpretation Codes → our condition labels
_WMO_MAP: dict[int, str] = {
    0: "sunny",
    1: "sunny",   2: "cloudy",  3: "cloudy",
    45: "cloudy", 48: "cloudy",
    51: "rainy",  53: "rainy",  55: "rainy",
    56: "rainy",  57: "rainy",
    61: "rainy",  63: "rainy",  65: "rainy",
    71: "snowy",  73: "snowy",  75: "snowy",  77: "snowy",
    80: "rainy",  81: "rainy",  82: "rainy",
    85: "snowy",  86: "snowy",
    95: "stormy", 96: "stormy", 99: "stormy",
}

CONDITION_EMOJI = {
    "sunny":  "☀️",
    "cloudy": "☁️",
    "rainy":  "🌧️",
    "snowy":  "❄️",
    "stormy": "⛈️",
}


def load_config() -> dict:
    try:
        with open(CONFIG_PATH) as fh:
            cfg = json.load(fh)
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        return cfg
    except FileNotFoundError:
        return DEFAULT_CONFIG.copy()


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as fh:
        json.dump(cfg, fh, indent=2)


def wmo_to_condition(code: int) -> str:
    return _WMO_MAP.get(code, "cloudy")


def fetch_forecast(lat: float, lon: float, forecast_days: int = 7) -> list[dict]:
    """
    Call Open-Meteo daily forecast endpoint.
    Returns list of dicts with date, temperature, temperature_max/min,
    precipitation, condition, wmo_code.
    Raises RuntimeError on network/parse failure.
    """
    params = urllib.parse.urlencode({
        "latitude":  lat,
        "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "weathercode",
            "windspeed_10m_max",
        ]),
        "timezone":      "auto",
        "forecast_days": forecast_days,
    })
    url = f"https://api.open-meteo.com/v1/forecast?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ForkCastAI/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read())
    except Exception as exc:
        raise RuntimeError(f"Open-Meteo request failed: {exc}") from exc

    daily = payload["daily"]
    results: list[dict] = []
    for i, dt_str in enumerate(daily["time"]):
        t_max  = daily["temperature_2m_max"][i] or 0.0
        t_min  = daily["temperature_2m_min"][i] or 0.0
        precip = daily["precipitation_sum"][i]  or 0.0
        prob   = daily.get("precipitation_probability_max", [None] * (i + 1))[i]
        wind   = daily.get("windspeed_10m_max",             [None] * (i + 1))[i]
        wmo    = daily["weathercode"][i] or 0

        results.append({
            "date":              date.fromisoformat(dt_str),
            "temperature":       round((t_max + t_min) / 2, 1),
            "temperature_max":   round(t_max, 1),
            "temperature_min":   round(t_min, 1),
            "precipitation":     round(precip, 1),
            "precipitation_prob": prob,
            "windspeed":         wind,
            "condition":         wmo_to_condition(wmo),
            "wmo_code":          wmo,
            "emoji":             CONDITION_EMOJI.get(wmo_to_condition(wmo), "🌤️"),
        })

    return results


def store_forecast(db: Session, forecast: list[dict]) -> int:
    """Upsert weather rows. Returns number of rows written."""
    count = 0
    for f in forecast:
        row = db.query(WeatherData).filter(WeatherData.weather_date == f["date"]).first()
        if row:
            row.temperature  = f["temperature"]
            row.precipitation = f["precipitation"]
            row.condition     = f["condition"]
        else:
            db.add(WeatherData(
                weather_date=f["date"],
                temperature=f["temperature"],
                precipitation=f["precipitation"],
                condition=f["condition"],
            ))
        count += 1
    db.commit()
    return count


def refresh_weather(db: Session) -> dict:
    """Load config → fetch forecast → store in DB. Returns structured result."""
    cfg      = load_config()
    lat, lon = cfg["latitude"], cfg["longitude"]
    forecast = fetch_forecast(lat, lon, forecast_days=7)
    stored   = store_forecast(db, forecast)

    return {
        "ok":       True,
        "location": {"lat": lat, "lon": lon, "name": cfg.get("restaurant_name", "")},
        "stored":   stored,
        "forecast": [
            {
                "date":              f["date"].isoformat(),
                "temperature":       f["temperature"],
                "temperature_max":   f["temperature_max"],
                "temperature_min":   f["temperature_min"],
                "precipitation":     f["precipitation"],
                "precipitation_prob": f["precipitation_prob"],
                "windspeed":         f["windspeed"],
                "condition":         f["condition"],
                "emoji":             f["emoji"],
            }
            for f in forecast
        ],
    }
