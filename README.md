![Uranus CI](https://github.com/petrobras-lps-ufrj-br/uranus/actions/workflows/ci.yml/badge.svg)
[![maestro](https://github.com/lps-ufrj-br/maestro-lightning/actions/workflows/flow.yml/badge.svg)](https://github.com/lps-ufrj-br/maestro-lighning/actions/workflows/flow.yml)

# 🪐 Uranus


Welcome to **Uranus**! This project is a modular framework designed for developing, training, and testing artificial intelligence models, with a strong focus on **Time Series Forecasting** using **PyTorch** and **PyTorch Lightning**.

## 🎯 Purpose

The goal of **Uranus** is to provide a clean, extensible structure for end-to-end AI workflows. Beyond model development, it serves as a comprehensive framework for orquestrating jobs (via **Airflow**) that collect data from **Cognite**, process it through high-performance inference servers like **NVIDIA Triton**, and visualize results in real-time. 

It abstracts away common boilerplate code for data loading, preprocessing, and training loops, allowing researchers and developers to focus on model architecture and feature engineering while maintaining a production-ready deployment path.

---

## 📂 Repository Structure

The codebase is organized into a main package named `uranus`, with infrastructure and orchestration handled by dedicated directories:

```text
.
├── 📂 uranus/              # Main project package
│   ├── 📂 ai/              # AI and Machine Learning core
│   │   ├── 📂 callbacks/   # Custom PyTorch Lightning callbacks
│   │   ├── 📂 clients/     # External API clients (e.g., Cognite)
│   │   ├── 📂 evaluation/  # Metrics and model evaluation tools
│   │   ├── 📂 loaders/     # Custom DataLoaders (e.g., Time Series Windowing)
│   │   ├── 📂 models/      # PyTorch Model architectures (e.g., MLP)
│   │   ├── 📂 preprocessing/ # Sktime & Sklearn pipelines
│   │   ├── 📂 runners/     # Scripts to execute training/inference jobs
│   │   ├── 📂 trainers/    # Training loops (Cross-Validation + Lightning)
│   │   └── 📂 visualization/ # Training and inference visualization tools
├── 📂 uranus/              # AI framework (Core logic, Trainers, Models)
├── 📂 servers/             # Infrastructure services (Postgres, Airflow, Triton, Grafana)
│   ├── 📂 workflows/       # Orchestration and automated workflows
│   │   ├── 📂 airflow/     # Airflow DAGs, logs, and plugins
│   │   └── 📂 model_repository/ # Triton Model assets and configs
│   ├── 📜 docker-compose.yml
│   └── 📜 README.md        # Infrastructure documentation
├── 📂 notebooks/           # Jupyter Notebooks for exploration and demos
├── 📂 scripts/             # Helper scripts
├── 📂 data/                # Raw data files (e.g., compressor.csv)
├── 📜 activate.sh          # Environment setup script
├── 📜 Makefile             # Shortcuts for orchestration and environment
├── 📜 docker-compose.yml   # MLflow tracking server
├── 📜 Dockerfile.mlflow    # Custom MLflow image with Postgres support
└── 📜 requirements.txt     # Python dependencies
```

---

## ✨ Key Features

*   **⚡ PyTorch Lightning Integration**: Robust training loops with built-in logging, checkpointing, and GPU support.
*   **🔄 Automated Cross-Validation**: The `Time Series Trainer` handles CV splits (e.g., TimeSeriesSplit) automatically and aggregates metrics.
*   **📊 Rich Logging & Evaluation**: 
    *   Beautiful ASCII metric tables and emoji-enhanced logs.
    *   Automatic collection of training and validation loss history.
    *   Custom `ModelCheckpoint` that saves model state, weights, and detailed history.
*   **🛠 Advanced Preprocessing**: Modular pipelines using `sktime` and `sklearn` for easy feature engineering (e.g., Lag features, Standard Scaling).
*   **🧵 Custom DataLoaders**: Flexible loaders that accept raw dataframes and handle windowing and batching on-the-fly.
*   **🧪 CI Test**: Automatic validation of module imports and package integrity on every push.

---

## ⚙️ Environment Configuration

The `activate.sh` script manages several environment variables required for both local development and Docker orchestration. These are automatically loaded when you run `make` or `source activate.sh`.

| Variable | Description | Default Value |
| :--- | :--- | :--- |
| `VIRTUALENV_NAMESPACE`| Name of the virtualenv directory | `.uranus-env` |
| `LOGURU_LEVEL` | Logging verbosity level | `DEBUG` |
| `URANUS_DATA_PATH` | Path to the raw data directory | `./data` |
| `MLFLOW_PORT` | Host port for the MLflow server | `8000` |
| `MLFLOW_TRACKING_URI` | URL of the MLflow tracking server | `http://localhost:8000` |
| `MLFLOW_ARTIFACT_PATH`| Directory for storing model artifacts | `./mlartifacts` |
| `MLFLOW_DB_PATH` | Path to the local SQLite database | `./mlflow.db` |

---

## 🚀 Getting Started

### Prerequisites

*   **OS**: Mac/Linux
*   **Tools**: `make`, `python3`, `virtualenv`

### Installation

This command will source `activate.sh`, create a virtual environment in `.uranus-env` (if it doesn't exist), and install the required packages.

### 🛠 Makefile Shortcuts

| Command | Description |
| :--- | :--- |
| `make` | Initialize environment and install dependencies |
| `make jupyter` | Launch Jupyter Lab instance |
| `make mlflow-up` | Start MLflow server (Port 5000) and its database |
| `make mlflow-down` | Stop and remove MLflow containers |
| `make clean` | Remove virtual environments and build artifacts |

### 📓 Running Notebooks

To launch a Jupyter Lab instance with the environment pre-configured:

```bash
make jupyter
```

---

## 🏃 Running Training Jobs

The repository provides a versatile script `scripts/job_v1.py` to run training jobs. It supports command-line arguments and JSON configuration files.

### Command Line Arguments

| Argument | Flag | Description | Default |
| :--- | :--- | :--- | :--- |
| `csv_path` | `--path`, `-p` | Path to the input CSV file (Required) | `None` |
| `fold` | `--fold`, `-f` | Specific fold index to train (Optional). If not set, trains all folds. | `None` |
| `epochs` | `--epochs`, `-e` | Number of training epochs | `20` |
| `splits` | `--splits`, `-s` | Number of Time Series Cross-Validation splits | `10` |
| `job_json` | `--job_json`, `-j` | Path to a JSON configuration file (Optional) | `None` |

### Examples

**1. Basic Run:**
Train on all folds using a specific CSV file.

```bash
python3 scripts/job_v1.py -p data/compressor.csv
```

**2. Train Specific Fold:**
Train only the 3rd fold with 50 epochs.

```bash
python3 scripts/job_v1.py -p data/compressor.csv -f 3 -e 50
```

**3. Custom Splits:**
Train with 5 cross-validation splits.

```bash
python3 scripts/job_v1.py -p data/compressor.csv -s 5
```

**4. Using a JSON Config:**
You can define your job parameters in a `job.json` file:

```json
{
    "csv_path": "data/dataset.csv",
    "fold": 0,
    "epochs": 100,
    "splits": 5
}
```

Then run it:

```bash
python3 scripts/job_v1.py -j job.json
```

---

## 📈 MLflow Tracking

**Uranus** uses **MLflow** for experiment tracking, model versioning, and artifact storage.

### 🚀 Starting the Server

To start the local MLflow server (runs in the background using SQLite):

```bash
make mlflow-up
```

*   **UI Access**: [http://localhost:8000](http://localhost:8000)
*   **Backend**: SQLite (stored in `mlflow.db`)
*   **Artifacts**: Locally stored in `mlartifacts/`
*   **Logs**: Server output is piped to `mlflow_server.log`

To stop the background server:

```bash
make mlflow-down
```

### 🛠 Setup & Usage

1.  **Initialize Environment**:
    ```bash
    source activate.sh
    ```
2.  **Start Services**:
    ```bash
    make mlflow-up
    ```
3.  **Python Integration**:
    No manual setup is needed in your Python scripts. Simply import `mlflow` and it will use the tracking URI from your environment:
    ```python
    import mlflow
    
    with mlflow.start_run():
        mlflow.log_param("sample_rate", 44100)
        mlflow.log_metric("accuracy", 0.95)
    ```

---

## 🏗 Infrastructure & Servers

The Uranus ecosystem relies on several external services for data storage, orchestration, and inference. These are managed within the `servers/` directory.

For detailed information on **Airflow**, **NVIDIA Triton**, **PostgreSQL**, and **monitoring tools**, please refer to:
👉 [**servers/README.md**](servers/README.md)

---

## 🛠 Python Usage Example

Here is a simplified example of how to set up a training pipeline programmatically using the `uranus` modules:

```python
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sktime.transformations.compose import TransformerPipeline
from uranus.ai.loaders import DataLoader_v1
from uranus.ai.models import MLP_v1
from uranus.ai.trainers.time_series import Trainer
from uranus.ai.preprocessing import Lag

# 1. Define Data & Features
features = {
    "input_1": "Raw_Sensor_1", 
    "target": "Raw_Target"
}
input_names = ["input_1"]
lags = {
    "input_1": Lag(10), 
    "target": Lag(-1)
}
preprocessors = {
    "input_1": TransformerPipeline([("scaler", StandardScaler())]),
    "target": TransformerPipeline([("scaler", StandardScaler())])
}

# 2. Initialize DataLoader
dataset = DataLoader_v1(
    path="data.csv",
    features=features,
    input_features=input_names,
    target_feature="target",
    lags=lags, 
    preprocessors=preprocessors
)

# 3. Setup Model & Trainer
model = MLP_v1(dataset=dataset, n_hidden=32)
trainer = Trainer(
    model=model,
    cv_strategy=TimeSeriesSplit(n_splits=4),
    accelerator='auto'
)

# 4. Train
# Returns a list of results (metrics & history) for each fold
trainer.fit(dataset, num_epochs=10)

# 5. Access History
print(results[0].history['val_loss'])
```

---

## 🤝 Contributing

Feel free to open issues or submit pull requests. Ensure all new modules have appropriate unit tests and follow the existing directory structure.
