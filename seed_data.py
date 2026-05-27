"""
Generates one full year of realistic synthetic restaurant data.
Patterns embedded: weekly cycles, seasonality, weather effects,
holidays, promotions, item-specific demand curves.
"""
import random
import math
from datetime import date, timedelta
from database import SessionLocal, MenuItem, Ingredient, BillOfMaterials, Sale, WeatherData, Holiday, Promotion

random.seed(42)

# ── Menu items ────────────────────────────────────────────────────────────────
MENU_ITEMS = [
    {"name": "Margherita Pizza",    "category": "main",      "price": 12.99, "base": 35},
    {"name": "BBQ Chicken Pizza",   "category": "main",      "price": 14.99, "base": 28},
    {"name": "Classic Beef Burger", "category": "main",      "price": 13.99, "base": 40},
    {"name": "Veggie Burger",       "category": "main",      "price": 11.99, "base": 18},
    {"name": "Caesar Salad",        "category": "appetizer", "price":  9.99, "base": 22},
    {"name": "Garden Salad",        "category": "appetizer", "price":  8.99, "base": 15},
    {"name": "Spaghetti Bolognese", "category": "main",      "price": 13.99, "base": 25},
    {"name": "Chicken Alfredo",     "category": "main",      "price": 14.99, "base": 20},
    {"name": "Fish & Chips",        "category": "main",      "price": 15.99, "base": 18},
    {"name": "Grilled Salmon",      "category": "main",      "price": 18.99, "base": 14},
    {"name": "Chicken Wings",       "category": "appetizer", "price": 12.99, "base": 30},
    {"name": "Garlic Bread",        "category": "appetizer", "price":  4.99, "base": 45},
    {"name": "Tiramisu",            "category": "dessert",   "price":  6.99, "base": 20},
    {"name": "Chocolate Lava Cake", "category": "dessert",   "price":  7.99, "base": 16},
    {"name": "Lemonade Pitcher",    "category": "beverage",  "price":  5.99, "base": 25},
]

# ── Ingredients ───────────────────────────────────────────────────────────────
INGREDIENTS = [
    # (name, unit, cost_per_unit, shelf_life_days, min_order_qty)
    ("Pizza Dough",       "kg",     1.20, 3,   5.0),
    ("Tomato Sauce",      "liter",  2.50, 7,   2.0),
    ("Mozzarella",        "kg",     8.00, 7,   2.0),
    ("Chicken Breast",    "kg",     7.50, 3,   3.0),
    ("Beef Patty (200g)", "piece",  2.80, 2,  20.0),
    ("Ground Beef",       "kg",     6.50, 2,   3.0),
    ("Lettuce",           "kg",     2.00, 3,   2.0),
    ("Tomato",            "kg",     1.80, 5,   3.0),
    ("Burger Bun",        "piece",  0.40, 3,  24.0),
    ("Spaghetti Pasta",   "kg",     1.50, 365, 5.0),
    ("Salmon Fillet",     "kg",    14.00, 2,   2.0),
    ("Cod Fillet",        "kg",     9.00, 2,   2.0),
    ("Potato",            "kg",     0.80, 30,  5.0),
    ("Garlic",            "kg",     4.00, 14,  1.0),
    ("Butter",            "kg",     6.00, 30,  2.0),
    ("Heavy Cream",       "liter",  3.50, 7,   2.0),
    ("Eggs",              "dozen",  3.00, 14,  2.0),
    ("Parmesan",          "kg",    12.00, 60,  1.0),
    ("Caesar Dressing",   "liter",  5.00, 14,  1.0),
    ("BBQ Sauce",         "liter",  3.00, 90,  1.0),
    ("Mixed Greens",      "kg",     4.00, 3,   2.0),
    ("Cocoa Powder",      "kg",     8.00, 365, 1.0),
    ("Mascarpone",        "kg",    10.00, 14,  1.0),
    ("Dark Chocolate",    "kg",    12.00, 365, 1.0),
    ("Lemon",             "kg",     2.00, 14,  3.0),
    ("Breadcrumbs",       "kg",     1.50, 30,  2.0),
    ("Flour",             "kg",     0.80, 365,10.0),
    ("Olive Oil",         "liter",  7.00, 365, 2.0),
    ("Chicken Wings",     "kg",     5.50, 2,   4.0),
]

# BOM: (menu_item_name, ingredient_name, qty_per_serving)
BOM = [
    # Margherita Pizza
    ("Margherita Pizza",    "Pizza Dough",       0.30),
    ("Margherita Pizza",    "Tomato Sauce",      0.10),
    ("Margherita Pizza",    "Mozzarella",        0.15),
    ("Margherita Pizza",    "Olive Oil",         0.02),
    # BBQ Chicken Pizza
    ("BBQ Chicken Pizza",   "Pizza Dough",       0.30),
    ("BBQ Chicken Pizza",   "BBQ Sauce",         0.08),
    ("BBQ Chicken Pizza",   "Mozzarella",        0.12),
    ("BBQ Chicken Pizza",   "Chicken Breast",    0.15),
    # Classic Beef Burger
    ("Classic Beef Burger", "Beef Patty (200g)", 1.00),
    ("Classic Beef Burger", "Burger Bun",        1.00),
    ("Classic Beef Burger", "Lettuce",           0.05),
    ("Classic Beef Burger", "Tomato",            0.05),
    # Veggie Burger
    ("Veggie Burger",       "Burger Bun",        1.00),
    ("Veggie Burger",       "Lettuce",           0.08),
    ("Veggie Burger",       "Tomato",            0.08),
    ("Veggie Burger",       "Mixed Greens",      0.05),
    # Caesar Salad
    ("Caesar Salad",        "Lettuce",           0.15),
    ("Caesar Salad",        "Caesar Dressing",   0.06),
    ("Caesar Salad",        "Parmesan",          0.03),
    ("Caesar Salad",        "Breadcrumbs",       0.02),
    # Garden Salad
    ("Garden Salad",        "Mixed Greens",      0.15),
    ("Garden Salad",        "Tomato",            0.08),
    ("Garden Salad",        "Olive Oil",         0.02),
    # Spaghetti Bolognese
    ("Spaghetti Bolognese", "Spaghetti Pasta",   0.15),
    ("Spaghetti Bolognese", "Ground Beef",       0.12),
    ("Spaghetti Bolognese", "Tomato Sauce",      0.10),
    ("Spaghetti Bolognese", "Parmesan",          0.02),
    ("Spaghetti Bolognese", "Olive Oil",         0.02),
    # Chicken Alfredo
    ("Chicken Alfredo",     "Spaghetti Pasta",   0.15),
    ("Chicken Alfredo",     "Chicken Breast",    0.15),
    ("Chicken Alfredo",     "Heavy Cream",       0.12),
    ("Chicken Alfredo",     "Parmesan",          0.04),
    ("Chicken Alfredo",     "Butter",            0.02),
    # Fish & Chips
    ("Fish & Chips",        "Cod Fillet",        0.20),
    ("Fish & Chips",        "Potato",            0.25),
    ("Fish & Chips",        "Flour",             0.05),
    ("Fish & Chips",        "Eggs",              0.08),
    ("Fish & Chips",        "Breadcrumbs",       0.05),
    # Grilled Salmon
    ("Grilled Salmon",      "Salmon Fillet",     0.22),
    ("Grilled Salmon",      "Potato",            0.15),
    ("Grilled Salmon",      "Butter",            0.02),
    ("Grilled Salmon",      "Garlic",            0.01),
    ("Grilled Salmon",      "Olive Oil",         0.02),
    # Chicken Wings
    ("Chicken Wings",       "Chicken Wings",     0.35),
    ("Chicken Wings",       "BBQ Sauce",         0.06),
    ("Chicken Wings",       "Garlic",            0.01),
    # Garlic Bread
    ("Garlic Bread",        "Burger Bun",        1.00),
    ("Garlic Bread",        "Butter",            0.04),
    ("Garlic Bread",        "Garlic",            0.02),
    # Tiramisu
    ("Tiramisu",            "Mascarpone",        0.10),
    ("Tiramisu",            "Eggs",              0.10),
    ("Tiramisu",            "Cocoa Powder",      0.02),
    ("Tiramisu",            "Heavy Cream",       0.05),
    # Chocolate Lava Cake
    ("Chocolate Lava Cake", "Dark Chocolate",    0.08),
    ("Chocolate Lava Cake", "Butter",            0.04),
    ("Chocolate Lava Cake", "Eggs",              0.12),
    ("Chocolate Lava Cake", "Flour",             0.03),
    # Lemonade Pitcher
    ("Lemonade Pitcher",    "Lemon",             0.15),
]

# US holidays 2024
HOLIDAYS_2024 = [
    (date(2024, 1, 1),  "New Year's Day",    0.0),   # closed
    (date(2024, 1, 15), "MLK Day",           0.9),
    (date(2024, 2, 14), "Valentine's Day",   1.85),
    (date(2024, 2, 19), "Presidents Day",    0.85),
    (date(2024, 3, 31), "Easter",            1.4),
    (date(2024, 5, 12), "Mother's Day",      1.65),
    (date(2024, 5, 27), "Memorial Day",      1.2),
    (date(2024, 7, 4),  "Independence Day",  1.3),
    (date(2024, 9, 2),  "Labor Day",         1.15),
    (date(2024, 11, 28),"Thanksgiving",      0.0),   # closed
    (date(2024, 11, 29),"Black Friday",      0.7),
    (date(2024, 12, 24),"Christmas Eve",     1.8),
    (date(2024, 12, 25),"Christmas Day",     0.0),   # closed
    (date(2024, 12, 31),"New Year's Eve",    2.0),
]

# Weekly demand multipliers (Mon=0 … Sun=6)
WEEKDAY_FACTOR = [0.72, 0.75, 0.88, 0.92, 1.30, 1.55, 1.20]

# Monthly seasonality multiplier (index 0 = Jan)
MONTH_FACTOR = [0.78, 0.80, 0.90, 0.95, 1.05, 1.12, 1.18, 1.15, 1.00, 0.95, 0.88, 1.25]

# Item-level seasonal bias: (summer_factor, winter_factor)
ITEM_SEASON = {
    "Caesar Salad":        (1.4, 0.7),
    "Garden Salad":        (1.5, 0.6),
    "Lemonade Pitcher":    (1.8, 0.4),
    "Spaghetti Bolognese": (0.8, 1.3),
    "Chicken Alfredo":     (0.8, 1.25),
    "Grilled Salmon":      (1.2, 0.9),
}


def _seasonal_item_factor(item_name: str, month: int) -> float:
    if item_name not in ITEM_SEASON:
        return 1.0
    summer_f, winter_f = ITEM_SEASON[item_name]
    # Interpolate between winter (Dec/Jan/Feb) and summer (Jun/Jul/Aug)
    summer_months = {6: 1.0, 7: 1.0, 8: 1.0, 5: 0.5, 9: 0.5}
    winter_months = {12: 1.0, 1: 1.0, 2: 1.0, 11: 0.5, 3: 0.5}
    sf = summer_months.get(month, 0.0)
    wf = winter_months.get(month, 0.0)
    base = 1.0
    return base + sf * (summer_f - 1.0) + wf * (winter_f - 1.0)


def _weather_factor(temp: float, precip: float, item_name: str) -> float:
    f = 1.0
    # Overall rain penalty
    if precip > 5:
        f *= 0.82
    elif precip > 1:
        f *= 0.92
    # Temperature effects
    if temp > 28:
        if "Salad" in item_name or "Lemonade" in item_name:
            f *= 1.30
        elif item_name in ("Spaghetti Bolognese", "Chicken Alfredo"):
            f *= 0.85
    elif temp < 5:
        if "Salad" in item_name or "Lemonade" in item_name:
            f *= 0.65
        elif item_name in ("Spaghetti Bolognese", "Chicken Alfredo"):
            f *= 1.25
    return f


def _generate_weather(start: date, days: int):
    """Synthetic weather for the UK/temperate climate."""
    records = []
    for i in range(days):
        d = start + timedelta(days=i)
        # Temperature: sinusoidal with noise
        day_of_year = d.timetuple().tm_yday
        base_temp = 10 + 10 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
        temp = base_temp + random.gauss(0, 3)
        # Precipitation: Poisson-ish
        rain_prob = 0.35 + 0.10 * math.cos(2 * math.pi * day_of_year / 365)
        precip = 0.0
        if random.random() < rain_prob:
            precip = random.expovariate(1 / 5)  # avg 5 mm
        condition = "sunny" if precip < 0.5 else ("rainy" if precip < 10 else "stormy")
        if temp < 0:
            condition = "snowy"
        records.append({"date": d, "temp": round(temp, 1), "precip": round(precip, 1), "condition": condition})
    return records


def seed_database():
    db = SessionLocal()
    try:
        if db.query(MenuItem).count() > 0:
            print("Database already seeded — skipping.")
            return

        print("Seeding database …")

        # ── Menu items ────────────────────────────────────────────────────────
        item_objs = {}
        for mi in MENU_ITEMS:
            obj = MenuItem(name=mi["name"], category=mi["category"],
                           base_price=mi["price"],
                           description=f"Freshly prepared {mi['name'].lower()}")
            db.add(obj)
            db.flush()
            item_objs[mi["name"]] = obj

        # ── Ingredients ───────────────────────────────────────────────────────
        ing_objs = {}
        for ing in INGREDIENTS:
            name, unit, cost, shelf, moq = ing
            obj = Ingredient(name=name, unit=unit, cost_per_unit=cost,
                             shelf_life_days=shelf, min_order_qty=moq,
                             current_stock=round(random.uniform(2, 15), 1))
            db.add(obj)
            db.flush()
            ing_objs[name] = obj

        # ── BOM ───────────────────────────────────────────────────────────────
        for item_name, ing_name, qty in BOM:
            db.add(BillOfMaterials(
                menu_item_id=item_objs[item_name].id,
                ingredient_id=ing_objs[ing_name].id,
                quantity=qty,
            ))

        # ── Holidays ──────────────────────────────────────────────────────────
        holiday_map = {}
        for hdate, hname, hfactor in HOLIDAYS_2024:
            db.add(Holiday(holiday_date=hdate, name=hname, impact_factor=hfactor))
            holiday_map[hdate] = hfactor

        # ── Promotions ────────────────────────────────────────────────────────
        burger_id = item_objs["Classic Beef Burger"].id
        veggie_id = item_objs["Veggie Burger"].id
        salad_ids = [item_objs["Caesar Salad"].id, item_objs["Garden Salad"].id]

        # Burger Tuesdays (all year)
        db.add(Promotion(menu_item_id=burger_id, name="Burger Tuesday",
                         start_date=date(2024, 1, 1), end_date=date(2024, 12, 31),
                         discount_pct=15.0))
        db.add(Promotion(menu_item_id=veggie_id, name="Burger Tuesday",
                         start_date=date(2024, 1, 1), end_date=date(2024, 12, 31),
                         discount_pct=15.0))
        # Summer salad special
        for sid in salad_ids:
            db.add(Promotion(menu_item_id=sid, name="Summer Salad Special",
                             start_date=date(2024, 6, 1), end_date=date(2024, 8, 31),
                             discount_pct=10.0))
        # Weekend dessert deal
        for dname in ("Tiramisu", "Chocolate Lava Cake"):
            db.add(Promotion(menu_item_id=item_objs[dname].id, name="Weekend Sweet Deal",
                             start_date=date(2024, 1, 1), end_date=date(2024, 12, 31),
                             discount_pct=20.0))

        db.commit()

        # Collect promotions into a lookup {item_id: [(start, end, pct, name)]}
        from collections import defaultdict
        promo_lookup = defaultdict(list)
        for p in db.query(Promotion).all():
            promo_lookup[p.menu_item_id].append((p.start_date, p.end_date, p.discount_pct, p.name))

        # ── Weather data ──────────────────────────────────────────────────────
        start_date = date(2024, 1, 1)
        num_days = 366
        weather_records = _generate_weather(start_date, num_days)
        weather_map = {}
        for wr in weather_records:
            obj = WeatherData(weather_date=wr["date"], temperature=wr["temp"],
                              precipitation=wr["precip"], condition=wr["condition"])
            db.add(obj)
            weather_map[wr["date"]] = wr
        db.commit()

        # ── Sales data ────────────────────────────────────────────────────────
        print("Generating 366 days of sales data …")
        sales_batch = []
        for day_offset in range(num_days):
            current_date = start_date + timedelta(days=day_offset)
            holiday_factor = holiday_map.get(current_date, 1.0)

            if holiday_factor == 0.0:
                continue  # restaurant closed

            wday = current_date.weekday()
            month = current_date.month
            wr = weather_map[current_date]

            day_factor = (WEEKDAY_FACTOR[wday]
                          * MONTH_FACTOR[month - 1]
                          * holiday_factor)

            for mi in MENU_ITEMS:
                item_obj = item_objs[mi["name"]]
                base_qty = mi["base"]

                # Seasonal item adjustment
                season_f = _seasonal_item_factor(mi["name"], month)
                weather_f = _weather_factor(wr["temp"], wr["precip"], mi["name"])

                # Promotion effect (only on correct day)
                promo_f = 1.0
                discount = 0.0
                promo_name = None
                for (ps, pe, pct, pname) in promo_lookup[item_obj.id]:
                    if ps <= current_date <= pe:
                        # Burger Tuesday: only Tuesdays
                        if "Tuesday" in pname and wday != 1:
                            continue
                        # Weekend deal: only Fri-Sun
                        if "Weekend" in pname and wday not in (4, 5, 6):
                            continue
                        promo_f = 1.0 + pct / 100 * 0.8  # 80% elasticity
                        discount = pct
                        promo_name = pname
                        break

                # Fish not popular on Mondays (freshness signal)
                if mi["name"] in ("Fish & Chips", "Grilled Salmon") and wday == 0:
                    weather_f *= 0.65

                computed = base_qty * day_factor * season_f * weather_f * promo_f
                # Poisson noise (count data)
                qty = max(0, int(random.gauss(computed, math.sqrt(computed) * 0.5)))

                if qty == 0:
                    continue

                sales_batch.append(Sale(
                    sale_date=current_date,
                    menu_item_id=item_obj.id,
                    quantity_sold=qty,
                    unit_price=mi["price"],
                    discount_pct=discount,
                ))

        db.bulk_save_objects(sales_batch)
        db.commit()
        print(f"Seeded {len(sales_batch)} sale records across {num_days} days.")

    finally:
        db.close()
