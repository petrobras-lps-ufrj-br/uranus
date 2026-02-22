__all__ = [
    "PI",
    "Product",
    "System",
    "Subsystem",
    "Sensor",
    "generate_id", 
    "DRIVER_OPTIONS"
]

import uuid
import json
from typing import Dict, Any
from sqlalchemy import Column, Integer, String, ForeignKey, JSON
from .database import Base

def generate_id():
    return str(uuid.uuid4())[:8]

DRIVER_OPTIONS = ["cognite", "scada", "generic"]

class PI(Base):
    __tablename__  = "pi"
    id             = Column(Integer, primary_key=True, index=True)
    pi_id          = Column(String, unique=True)
    name           = Column(String, unique=True)
    driver         = Column(String, default="generic") # cognite, scada, generic
    description    = Column(String)
    _configuration = Column("configuration", JSON, default={})

    @property
    def configuration(self) -> Dict[str, Any]:
        if isinstance(self._configuration, str):
            return json.loads(self._configuration)
        return self._configuration or {}

    @configuration.setter
    def configuration(self, value: Dict[str, Any]):
        self._configuration = value

    def __getitem__(self, key: str) -> Any:
        conf = self.configuration
        if key not in conf:
            raise KeyError(f"Key {key} not found in configuration")
        return conf[key]

    def __setitem__(self, key: str, value: Any):
        conf = self.configuration
        conf[key] = value
        self.configuration = conf

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
    _parents        = Column("parents", JSON, default=[]) # List of subsystem_id strings
    _children       = Column("children", JSON, default=[]) # List of subsystem_id strings

    @property
    def parents(self):
        if isinstance(self._parents, str):
            return json.loads(self._parents)
        return self._parents or []

    @property
    def children(self):
        if isinstance(self._children, str):
            return json.loads(self._children)
        return self._children or []

    @parents.setter
    def parents(self, value):
        self._parents = value

    @children.setter
    def children(self, value):
        self._children = value

class Sensor(Base):
    __tablename__   = "sensors"
    id              = Column(Integer, primary_key=True, index=True)
    sensor_id       = Column(String, unique=True)
    subsystem_id    = Column(String, ForeignKey("subsystems.subsystem_id", ondelete="CASCADE"))
    pi_id           = Column(String, ForeignKey("pi.pi_id", ondelete="CASCADE"), nullable=True)
    name            = Column(String, index=True)
    description     = Column(String)
    tag             = Column(String)