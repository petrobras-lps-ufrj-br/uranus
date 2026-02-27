
from uuid import uuid4
from fastapi import FastAPI, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Generator
from uranus.servers.acquisition.models import db_schemas, database
from uranus.servers.acquisition import drivers
from uranus.servers import schemas

from fastapi.middleware.cors import CORSMiddleware

def generate_id():
    return str(uuid4())[:8]

# Create tables if they don't exist
db_schemas.Base.metadata.create_all(bind=database.engine)

app = FastAPI(
    title="Uranus Acquisition API",
    description="""
    This API manages the Process Instruments (PI) used for data acquisition.
    It allows for registering, updating, and deleting PIs, as well as listing available drivers.
    """,
    version="1.0.0",
    docs_url="/manual",
    redoc_url="/redoc",
    contact={
        "name": "Uranus Team",
        "url": "https://github.com/petrobras-lps-ufrj-br/uranus",
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get DB session
def get_db() -> Generator[Session, None, None]:
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", tags=["General"], summary="API Root")
def read_root() -> Dict[str, str]:
    return {
        "name": "Uranus Acquisition API",
        "version": "1.0.0",
        "docs": "/manual",
        "status": "online"
    }

@app.get("/drivers", response_model=List[str], tags=["Drivers"], summary="List available drivers")
def list_drivers() -> List[str]:
    """
    Returns a list of all supported acquisition drivers.
    """
    return db_schemas.DRIVER_OPTIONS

@app.get("/pis", response_model=List[schemas.PI], tags=["PI Management"], summary="List all PI information")
def list_pis(db: Session = Depends(get_db)) -> List[schemas.PI]:
    """
    Retrieve all registered Process Instruments (PIs) from the database.
    """
    pis = db.query(db_schemas.PI).all()
    return [schemas.PI.model_validate(pi) for pi in pis]

@app.post("/pis", response_model=schemas.PI, status_code=status.HTTP_201_CREATED, tags=["PI Management"], summary="Create a new PI")
def create_pi(pi: schemas.PI, db: Session = Depends(get_db)) -> schemas.PI:
    """
    Register a new Process Instrument.
    
    - **name**: Human-readable name
    - **pi_id**: Unique identifier
    - **driver**: One of the supported drivers (cognite, scada, etc.)
    - **description**: Optional description
    - **params**: Driver-specific configuration
    """
   
    new_pi = db_schemas.PI(
        pi_id=generate_id(),
        name=pi.name,
        driver=pi.driver,
        description=pi.description,
        params=pi.params
    )
    db.add(new_pi)
    db.commit()
    db.refresh(new_pi)
    return schemas.PI.model_validate(new_pi)

@app.get("/pis/{pi_id}", response_model=schemas.PI, tags=["PI Management"], summary="Get PI information")
def get_pi(pi_id: str, db: Session = Depends(get_db)) -> schemas.PI:
    """
    Retrieve detailed information for a specific Process Instrument by its ID.
    """
    db_pi = db.query(db_schemas.PI).filter(db_schemas.PI.pi_id == pi_id).first()
    if not db_pi:
        raise HTTPException(status_code=404, detail="PI not found")
    return schemas.PI.model_validate(db_pi)

@app.put("/pis/{pi_id}", response_model=schemas.PI, tags=["PI Management"], summary="Edit a PI")
def update_pi(pi_id: str, pi_update: schemas.PIUpdate, db: Session = Depends(get_db)) -> schemas.PI:
    """
    Update an existing Process Instrument's information.
    """
    db_pi = db.query(db_schemas.PI).filter(db_schemas.PI.pi_id == pi_id).first()
    if not db_pi:
        raise HTTPException(status_code=404, detail="PI not found")
    
    update_data = pi_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_pi, key, value)
    
    db.commit()
    db.refresh(db_pi)
    return schemas.PI.model_validate(db_pi)

@app.delete("/pis/{pi_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["PI Management"], summary="Delete a PI")
def delete_pi(pi_id: str, db: Session = Depends(get_db)) -> None:
    """
    Remove a Process Instrument from the system.
    """
    db_pi = db.query(db_schemas.PI).filter(db_schemas.PI.pi_id == pi_id).first()
    if not db_pi:
        raise HTTPException(status_code=404, detail="PI not found")
    
    db.delete(db_pi)
    db.commit()
    return None

@app.post("/pis/{pi_id}/data", response_model=schemas.DataResponse, tags=["Data Acquisition"], summary="Retrieve data from PI")
def get_pi_data(pi_id: str, time_range: schemas.TimeRange, db: Session = Depends(get_db)) -> schemas.DataResponse:
    """
    Fetch time series data from the specified Process Instrument.
    
    Currently supports:
    - **Cognite**: Requires `client_secret` and `tenant_id` in the PI configuration parameters.
    """
    db_pi = db.query(db_schemas.PI).filter(db_schemas.PI.pi_id == pi_id).first()
    if not db_pi:
        raise HTTPException(status_code=404, detail=f"PI with ID '{pi_id}' not found")
    
    if db_pi.driver == "cognite":
        try:
            # Instantiate Cognite driver with PI parameters
            params = db_pi.params
            cognite_driver = drivers.Cognite(
                name=db_pi.name,
                client_secret=params.get("client_secret"),
                tenant_id=params.get("tenant_id"),
                project=params.get("project", "publicdata"),
                client_id=params.get("client_id", "1b90ede3-271e-401b-81a0-a4d52bea3273")
            )
            
            # Fetch data
            df_data, df_info = cognite_driver.get_dataframe(
                asset_name=time_range.asset_name,
                start_time=time_range.start_time,
                end_time=time_range.end_time,
                granularity=time_range.granularity
            )
            
            # Convert DataFrames to JSON-serializable formats
            # data: records format (list of dicts)
            # info: to_dict() for the metadata dataframe
            return schemas.DataResponse(
                data=df_data.reset_index().to_dict(orient="records"),
                info=df_info.to_dict(orient="records")[0] if not df_info.empty else {}
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cognite Error: {str(e)}")
            
    raise HTTPException(
        status_code=501, 
        detail=f"Driver '{db_pi.driver}' is not yet implemented for data retrieval."
    )


@app.get("/pis/{pi_id}/metrics", response_model=List[str], tags=["Data Acquisition"], summary="List available metrics for an asset")
def list_pi_metrics(pi_id: str, asset_name: str, db: Session = Depends(get_db)) -> List[str]:
    """
    List all available time series names for a specific asset associated with a PI.
    """
    db_pi = db.query(db_schemas.PI).filter(db_schemas.PI.pi_id == pi_id).first()
    if not db_pi:
        raise HTTPException(status_code=404, detail=f"PI with ID '{pi_id}' not found")
    
    if db_pi.driver == "cognite":
        try:
            params = db_pi.params
            cognite_driver = drivers.Cognite(
                name=db_pi.name,
                client_secret=params.get("client_secret"),
                tenant_id=params.get("tenant_id"),
                project=params.get("project", "publicdata"),
                client_id=params.get("client_id", "1b90ede3-271e-401b-81a0-a4d52bea3273")
            )
            return cognite_driver.get_time_series_names(asset_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cognite Error: {str(e)}")
    
    raise HTTPException(
        status_code=501, 
        detail=f"Driver '{db_pi.driver}' does not support metric listing."
    )


@app.get("/pis/{pi_id}/metrics/range", response_model=List[schemas.MetricRange], tags=["Data Acquisition"], summary="List metrics range for an asset")
def list_pi_metrics_range(pi_id: str, asset_name: str, db: Session = Depends(get_db)) -> List[schemas.MetricRange]:
    """
    List all available time series and their data range (first/last timestamp) for a specific asset.
    """
    db_pi = db.query(db_schemas.PI).filter(db_schemas.PI.pi_id == pi_id).first()
    if not db_pi:
        raise HTTPException(status_code=404, detail=f"PI with ID '{pi_id}' not found")
    
    if db_pi.driver == "cognite":
        try:
            params = db_pi.params
            cognite_driver = drivers.Cognite(
                name=db_pi.name,
                client_secret=params.get("client_secret"),
                tenant_id=params.get("tenant_id"),
                project=params.get("project", "publicdata"),
                client_id=params.get("client_id", "1b90ede3-271e-401b-81a0-a4d52bea3273")
            )
            df_range = cognite_driver.get_time_series_range(asset_name)
            return df_range.to_dict(orient="records")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cognite Error: {str(e)}")
    
    raise HTTPException(
        status_code=501, 
        detail=f"Driver '{db_pi.driver}' does not support metric range listing."
    )




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
