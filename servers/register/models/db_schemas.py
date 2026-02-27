__all__ = [
    "Product",
    "System",
    "Subsystem",
    "Sensor",
]

import json
from typing import Dict, Any
from sqlalchemy import Column, Integer, String, ForeignKey, JSON
from .database import Base



class Product(Base):
    __tablename__ = "product"
    id            = Column(Integer, primary_key=True, index=True)
    product_id    = Column(String, unique=True)
    name          = Column(String, unique=True)
    description   = Column(String)
    location_type = Column(String) # on-shore or off-shore
    country       = Column(String)
    state         = Column(String)
    number        = Column(String)

class System(Base):
    __tablename__   = "systems"
    id              = Column(Integer, primary_key=True, index=True)
    system_id       = Column(String, unique=True)
    product_id      = Column(String, ForeignKey("product.product_id", ondelete="CASCADE"))
    name            = Column(String, unique=True)
    description     = Column(String)

class Subsystem(Base):
    __tablename__   = "subsystems"
    id              = Column(Integer, primary_key=True, index=True)
    subsystem_id    = Column(String, unique=True)
    system_id       = Column(String, ForeignKey("systems.system_id", ondelete="CASCADE"))
    name            = Column(String, unique=True)
    description     = Column(String)


class Sensor(Base):
    __tablename__   = "sensors"
    id              = Column(Integer, primary_key=True, index=True)
    sensor_id       = Column(String, unique=True)
    subsystem_id    = Column(String, ForeignKey("subsystems.subsystem_id", ondelete="CASCADE"))
    pi_id           = Column(String, ForeignKey("pi.pi_id", ondelete="CASCADE"), nullable=True)
    name            = Column(String, index=True)
    description     = Column(String)
    tag             = Column(String)