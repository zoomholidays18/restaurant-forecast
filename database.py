from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    Date, DateTime, ForeignKey, Boolean, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

DATABASE_URL = "sqlite:///./restaurant.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class MenuItem(Base):
    __tablename__ = "menu_items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)  # appetizer, main, dessert, beverage
    description = Column(Text)
    base_price = Column(Float, nullable=False)
    is_active = Column(Boolean, default=True)

    sales = relationship("Sale", back_populates="menu_item")
    bom = relationship("BillOfMaterials", back_populates="menu_item")
    promotions = relationship("Promotion", back_populates="menu_item")
    predictions = relationship("Prediction", back_populates="menu_item")


class Ingredient(Base):
    __tablename__ = "ingredients"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    unit = Column(String, nullable=False)  # kg, liter, piece
    cost_per_unit = Column(Float, nullable=False)
    shelf_life_days = Column(Integer, default=7)
    min_order_qty = Column(Float, default=1.0)
    current_stock = Column(Float, default=0.0)

    bom = relationship("BillOfMaterials", back_populates="ingredient")


class BillOfMaterials(Base):
    __tablename__ = "bill_of_materials"
    id = Column(Integer, primary_key=True, index=True)
    menu_item_id = Column(Integer, ForeignKey("menu_items.id"), nullable=False)
    ingredient_id = Column(Integer, ForeignKey("ingredients.id"), nullable=False)
    quantity = Column(Float, nullable=False)

    menu_item = relationship("MenuItem", back_populates="bom")
    ingredient = relationship("Ingredient", back_populates="bom")


class Sale(Base):
    __tablename__ = "sales"
    id = Column(Integer, primary_key=True, index=True)
    sale_date = Column(Date, nullable=False, index=True)
    menu_item_id = Column(Integer, ForeignKey("menu_items.id"), nullable=False)
    quantity_sold = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)
    discount_pct = Column(Float, default=0.0)

    menu_item = relationship("MenuItem", back_populates="sales")


class WeatherData(Base):
    __tablename__ = "weather_data"
    id = Column(Integer, primary_key=True, index=True)
    weather_date = Column(Date, nullable=False, unique=True, index=True)
    temperature = Column(Float)
    precipitation = Column(Float, default=0.0)
    condition = Column(String)  # sunny, cloudy, rainy, snowy, stormy


class Holiday(Base):
    __tablename__ = "holidays"
    id = Column(Integer, primary_key=True, index=True)
    holiday_date = Column(Date, nullable=False, unique=True, index=True)
    name = Column(String, nullable=False)
    impact_factor = Column(Float, default=1.0)


class Promotion(Base):
    __tablename__ = "promotions"
    id = Column(Integer, primary_key=True, index=True)
    menu_item_id = Column(Integer, ForeignKey("menu_items.id"), nullable=True)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    discount_pct = Column(Float, default=0.0)
    name = Column(String, nullable=False)

    menu_item = relationship("MenuItem", back_populates="promotions")


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    prediction_date = Column(Date, nullable=False, index=True)
    menu_item_id = Column(Integer, ForeignKey("menu_items.id"), nullable=False)
    predicted_qty = Column(Float, nullable=False)
    lower_bound = Column(Float)
    upper_bound = Column(Float)
    actual_qty = Column(Integer, nullable=True)
    is_manual_override = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    menu_item = relationship("MenuItem", back_populates="predictions")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
