# 📟 Uranus Display Server

This project provides a FastAPI server with an integrated Dash dashboard for managing Products, Systems, Subsystems, and Sensors.

## Features
- **FastAPI**: Backend API for data management.
- **Dash Dashboard**: User-friendly UI for CRUD operations on hierarchical data.
- **PostgreSQL**: Robust database for persistent storage.
- **Docker Compose**: One-command setup for the entire stack (Postgres, pgAdmin, Grafana, App).

## Getting Started

### 1. Build and Start the stack
```bash
docker-compose up --build
```

### 2. Access the services
- **FastAPI/Dashboard**: [http://localhost:8000/dashboard/](http://localhost:8000/dashboard/)
- **pgAdmin**: [http://localhost:5050](http://localhost:5050) (Login: admin@admin.com / admin)
- **Grafana**: [http://localhost:3000](http://localhost:3000) (Login: admin / admin)

## Folder Structure
- `models/`: SQLAlchemy database models.
- `visual/`: Dash dashboard components and logic.
- `main.py`: Entry point for the FastAPI server.
- `docker-compose.yml`: Infrastructure configuration.
