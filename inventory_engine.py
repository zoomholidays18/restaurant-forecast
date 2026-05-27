"""
Inventory recommendation engine.

Steps:
1. Multiply predicted quantities by BOM quantities to get gross ingredient needs.
2. Add safety stock (proportional to prediction uncertainty × shelf-life penalty).
3. Subtract current stock.
4. Round up to supplier minimum order quantities.
5. Flag items at risk of stockout or over-stock.
"""
from __future__ import annotations
import math
from collections import defaultdict
from sqlalchemy.orm import Session
from database import BillOfMaterials, Ingredient, MenuItem

# Safety stock target: cover this fraction of extra demand beyond prediction
SAFETY_MULTIPLIER = 0.20       # 20 % buffer on top of predicted need
PERISHABLE_THRESHOLD = 5       # shelf_life_days ≤ this → apply perishability penalty
PERISHABLE_PENALTY   = 0.10    # cut safety stock by 10 % for very perishable items


class InventoryEngine:

    def _build_bom_map(self, db: Session) -> dict[int, list[tuple[int, float]]]:
        """Returns {menu_item_id: [(ingredient_id, qty_per_serving), ...]}"""
        bom_map = defaultdict(list)
        for row in db.query(BillOfMaterials).all():
            bom_map[row.menu_item_id].append((row.ingredient_id, row.quantity))
        return dict(bom_map)

    def _build_ingredient_map(self, db: Session) -> dict[int, Ingredient]:
        return {i.id: i for i in db.query(Ingredient).all()}

    def calculate_requirements(
        self,
        predictions: list[dict],
        db: Session,
    ) -> list[dict]:
        """
        Given a list of prediction dicts (from SalesForecaster.predict_next_day),
        return a recommendation list per ingredient.

        Each dict contains:
            ingredient_id, name, unit, cost_per_unit, shelf_life_days,
            current_stock, gross_required, safety_stock, total_required,
            to_order, order_units, estimated_cost,
            status (ok / low / critical)
        """
        bom_map  = self._build_bom_map(db)
        ing_map  = self._build_ingredient_map(db)

        # Accumulate ingredient needs
        gross:    dict[int, float] = defaultdict(float)
        max_std:  dict[int, float] = defaultdict(float)  # for safety-stock sizing

        for pred in predictions:
            item_id  = pred["item_id"]
            qty      = pred["predicted_qty"]
            std      = pred.get("std", 0.0)

            for (ing_id, qty_per_serving) in bom_map.get(item_id, []):
                gross[ing_id]   += qty * qty_per_serving
                max_std[ing_id] += std * qty_per_serving

        results = []
        for ing_id, ing in ing_map.items():
            gross_req = gross.get(ing_id, 0.0)
            if gross_req == 0.0:
                continue  # ingredient not used tomorrow

            # Safety stock
            base_safety = gross_req * SAFETY_MULTIPLIER
            # Reduce safety buffer for very perishable items to avoid waste
            if ing.shelf_life_days <= PERISHABLE_THRESHOLD:
                base_safety *= (1.0 - PERISHABLE_PENALTY)
            # Also scale safety by prediction uncertainty
            uncertainty_buffer = max_std.get(ing_id, 0.0) * 0.5
            safety_stock = max(base_safety, uncertainty_buffer)

            total_required = gross_req + safety_stock

            # Current stock deduction
            already_available = max(0.0, ing.current_stock)
            net_to_order = max(0.0, total_required - already_available)

            # Round up to minimum order quantity
            if net_to_order > 0:
                order_units = math.ceil(net_to_order / ing.min_order_qty) * ing.min_order_qty
            else:
                order_units = 0.0

            estimated_cost = round(order_units * ing.cost_per_unit, 2)

            # Status
            if already_available < gross_req * 0.5:
                status = "critical"
            elif already_available < gross_req:
                status = "low"
            else:
                status = "ok"

            results.append({
                "ingredient_id":  ing_id,
                "name":           ing.name,
                "unit":           ing.unit,
                "cost_per_unit":  ing.cost_per_unit,
                "shelf_life_days": ing.shelf_life_days,
                "current_stock":  round(already_available, 2),
                "gross_required": round(gross_req, 2),
                "safety_stock":   round(safety_stock, 2),
                "total_required": round(total_required, 2),
                "to_order":       round(net_to_order, 2),
                "order_units":    round(order_units, 2),
                "estimated_cost": estimated_cost,
                "status":         status,
            })

        results.sort(key=lambda x: (x["status"] != "critical", x["status"] != "low", x["name"]))
        return results

    def summary_metrics(
        self, predictions: list[dict], recommendations: list[dict]
    ) -> dict:
        total_predicted_orders = int(sum(p["predicted_qty"] for p in predictions))
        predicted_revenue = round(
            sum(p["predicted_qty"] * p["base_price"] * (1 - p["promo_discount"] / 100)
                for p in predictions), 2
        )
        total_order_cost = round(sum(r["estimated_cost"] for r in recommendations), 2)
        stockout_risks   = sum(1 for r in recommendations if r["status"] == "critical")
        items_to_order   = sum(1 for r in recommendations if r["order_units"] > 0)
        waste_saved_est  = round(
            sum(max(0.0, r["current_stock"] - r["total_required"]) * r["cost_per_unit"]
                for r in recommendations), 2
        )

        return {
            "total_predicted_orders": total_predicted_orders,
            "predicted_revenue":      predicted_revenue,
            "total_order_cost":       total_order_cost,
            "stockout_risks":         stockout_risks,
            "items_to_order":         items_to_order,
            "waste_savings_est":      waste_saved_est,
        }

    def update_stock(self, db: Session, ingredient_id: int, new_qty: float) -> Ingredient:
        ing = db.query(Ingredient).filter(Ingredient.id == ingredient_id).first()
        if ing:
            ing.current_stock = new_qty
            db.commit()
            db.refresh(ing)
        return ing
