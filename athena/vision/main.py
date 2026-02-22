from fastapi import FastAPI
import uvicorn
from models import Base, engine
from visual import create_dashboard
from fastapi.middleware.wsgi import WSGIMiddleware

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Cognite Open Data API")

# Create the Dash dashboard
dash_app = create_dashboard(requests_pathname_prefix="/dashboard/", server=True)

# Mount the Dash app
app.mount("/dashboard", WSGIMiddleware(dash_app.server))

@app.get("/")
def read_root():
    return {
        "message": "Cognite Open Data Server is running.",
        "endpoints": {
            "Dashboard": "/dashboard/",
            "API Docs": "/docs"
        }
    }

if __name__ == "__main__":
    # Start the server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
