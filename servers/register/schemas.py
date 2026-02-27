
__all__ = [
    "PI",
    "json_encode",
    "json_decode",
    "json_save",
    "json_load",
    "PIUpdate",
    "TimeRange",
    "DataResponse",
    "MetricRange"
    ]

import json
from datetime import datetime
from typing import Dict, Any, Union, List
from pydantic import BaseModel, Field, ConfigDict


encoder=json.JSONEncoder
decoder=json.JSONDecoder

def json_encode( obj : Any ) -> str:
    return json.dumps( obj, cls=encoder) 

def json_decode( obj : str ) -> Any:
    return json.loads( obj , cls=decoder)

def json_save( obj, f ):
   json.dump(obj, f, cls=encoder)

def json_load( f ):
   return json.load(f,  cls=decoder)


class PIUpdate(BaseModel):
    name: Union[str, None] = Field(None, description="The human-readable name of the Process Instrument", example="PI-101-Updated")
    driver: Union[str, None] = Field(None, description="The acquisition driver type", example="scada")
    description: Union[str, None] = Field(None, description="Detailed description of the instrument", example="Updated description")
    params: Union[Dict[str, Any], None] = Field(None, description="Driver-specific configuration parameters", example={"url": "http://scada.example.com"})

class PI(PIUpdate):
    name: str = Field(..., description="The name of the Process Instrument", example="PI-101")
    pi_id: str = Field(..., description="Unique identifier for the Process Instrument", example="PI_001")
    driver: str = Field(..., description="The driver type for the PI", example="cognite")
    model_config = ConfigDict(from_attributes=True)
    
class TimeRange(BaseModel):
    start_time: datetime = Field(..., description="Start time for data retrieval", example="2023-01-01T00:00:00")
    end_time: datetime = Field(..., description="End time for data retrieval", example="2023-01-02T00:00:00")
    asset_name: str = Field(..., description="Name of the asset in Cognite", example="Asset_01")
    granularity: str = Field("2s", description="Data granularity (e.g., '1m', '1h', '2s')", example="1m")

class DataResponse(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="The retrieved time series data")
    info: Dict[str, Any] = Field(..., description="Metadata information about the metrics")

class MetricRange(BaseModel):
    metric: str = Field(..., description="The name of the metric")
    external_id: str = Field(..., description="The external ID of the metric")
    first_timestamp: Union[datetime, None] = Field(None, description="The timestamp of the first data point")
    last_timestamp: Union[datetime, None] = Field(None, description="The timestamp of the last data point")
