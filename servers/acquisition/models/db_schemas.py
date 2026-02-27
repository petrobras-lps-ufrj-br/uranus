__all__ = [
    "PI",
    "generate_id", 
    "DRIVER_OPTIONS"
]

import uuid
import json
from typing import Dict, Any
from sqlalchemy import Column, Integer, String, ForeignKey, JSON
from .database import Base



DRIVER_OPTIONS = ["cognite", "scada", "prometheus", "generic"]

class PI(Base):

    __tablename__  = "pis"
    id             = Column(Integer, primary_key=True, index=True)
    pi_id          = Column(String, unique=True)
    name           = Column(String, unique=True)
    driver         = Column(String, default="generic") # cognite, scada, generic
    description    = Column(String)
    _params        = Column("params", JSON, default={})

    @property
    def params(self) -> Dict[str, Any]:
        if isinstance(self._params, str):
            return json.loads(self._params)
        return self._params or {}

    @params.setter
    def params(self, value: Dict[str, Any]):
        self._params= value

    def __getitem__(self, key: str) -> Any:
        conf = self.params
        if key not in conf:
            raise KeyError(f"Key {key} not found in params")
        return conf[key]

    def __setitem__(self, key: str, value: Any):
        conf = self.params
        conf[key] = value
        self.params= conf
