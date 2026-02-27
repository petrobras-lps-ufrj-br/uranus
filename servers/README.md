# 🗄️ Uranus Servers

This directory contains the infrastructure configuration for the Uranus ecosystem services, including data storage, observation, and orchestration.

## 🐳 Services Overview

The following core services are managed via the `docker-compose.yml` in this directory:

| Service | Technology | Port | Description |
| :--- | :--- | :--- | :--- |
| **Database** | PostgreSQL | 5432 | Main relational storage for application data |
| **Admin UI** | pgAdmin 4 | 5050 | Web interface for PostgreSQL management |
| **Time-Series**| InfluxDB | 8086 | High-performance storage for sensor metrics |
| **Visualization**| Grafana | 3000 | Dashboard for monitoring system health and metrics |
| **Orchestration**| Airflow | 8081 | Workflow management and job scheduling |
| **Inference** | NVIDIA Triton | 8000 | (Optional) High-performance model serving |

---

## 🌪 Airflow Orchestration

Airflow is used to schedule and monitor the end-to-end data pipelines.

*   **Access**: [http://localhost:8081](http://localhost:8081)
*   **Credentials**: `admin` / `admin`
*   **Database**: Uses a dedicated PostgreSQL instance (`uranus_airflow_db`).
*   **Volume Mapping**: 
    - DAGs: `./workflows/airflow/dags`
    - Logs: `./workflows/airflow/logs`
    - Plugins: `./workflows/airflow/plugins`

---

## 📡 Triton Inference Server

Uranus integrates with **NVIDIA Triton Inference Server** for high-performance model serving. 

### 🏗 Model Repository

Models are served from the `workflows/model_repository/` directory. The structure should follow:

```text
workflows/model_repository/
└── <model_name>/
    ├── config.pbtxt        # Model configuration
    └── 1/                  # Version number
        └── model.pt        # Model file
```

### 🚀 Usage

To enable Triton, uncomment the `triton` section in `docker-compose.yml`. It requires `nvidia-container-toolkit` for GPU support.

---

## 🏃 Running the External Servers

To start the full stack (Database, Grafana, InfluxDB, and Airflow):

```bash
cd servers
docker-compose up -d
```

To stop:
```bash
docker-compose down
```
