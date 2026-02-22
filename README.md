![Uranus CI](https://github.com/petrobras-lps-ufrj-br/uranus/actions/workflows/ci.yml/badge.svg)


# 🪐 Uranus


Welcome to **Uranus**! This project is a modular framework designed for developing, training, and testing artificial intelligence models, with a strong focus on **Time Series Forecasting** using **PyTorch** and **PyTorch Lightning**.

## 🎯 Purpose

The goal of **Uranus** is to provide a clean, extensible structure for end-to-end AI workflows. Beyond model development, it serves as a comprehensive framework for orquestrating jobs (via **Airflow**) that collect data from **Cognite**, process it through high-performance inference servers like **NVIDIA Triton**, and visualize results in real-time. 

It abstracts away common boilerplate code for data loading, preprocessing, and training loops, allowing researchers and developers to focus on model architecture and feature engineering while maintaining a production-ready deployment path.

---

## 📂 Repository Structure

The codebase is organized into a main package named `uranus`, with sub-packages for AI and visualization:

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
│   └── 📂 display/         # Visualization and dashboard tools
│
├── 📂 notebooks/           # Jupyter Notebooks for exploration and demos
├── 📂 scripts/             # Helper scripts
├── 📜 activate.sh          # Environment setup script
├── 📜 Makefile             # Shortcuts for installation and running
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

## 🚀 Getting Started

### Prerequisites

*   **OS**: Mac/Linux
*   **Tools**: `make`, `python3`, `virtualenv`

### Installation

To set up the environment and install dependencies, simply run:

```bash
make
```

This command will source `activate.sh`, create a virtual environment in `.uranus-env` (if it doesn't exist), and install the required packages.

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

## 📡 In-Production Inference (Triton)

**Uranus** integrates with **NVIDIA Triton Inference Server** for high-performance model serving. 

### 🏗 Model Repository

Models should be placed in the `data/model_repository/` directory following this schema:

```text
data/model_repository/
└── <model_name>/
    ├── config.pbtxt        # Model configuration
    └── 1/                  # Version number
        └── model.pt        # Model file
```

For more details, see [data/model_repository/README.md](data/model_repository/README.md).

### 🐳 Deployment

To start the full stack (Database, Grafana, InfluxDB, and Triton):

```bash
docker-compose up -d
```

---

## 🖥 Display & Visualization

The `uranus.display` module provides a powerful, configurable dashboard for real-time monitoring and data interaction.

### 📊 Key Capabilities

*   **📈 Time Series Visualization**: Interactive plots to monitor sensor data, model inputs, and forecasting results.
*   **🚨 Alarm System**: Configurable threshold-based alarms to notify users of anomalies or performance degradation.
*   **⚙️ Configurable Server**: A flexible FastAPI backend that serves as the integration point for data collection and visualization.
*   **🛠 Model Adaptation**: Tools to visualize model drift and facilitate manual or automated adjustments to the inference pipeline.

For detailed configuration, see the [uranus/display/README.md](uranus/display/README.md).

---

## 🛠 Python Usage Example

Here is a simplified example of how to set up a training pipeline programmatically using the `uranus` modules:

```python
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sktime.transformations.compose import TransformerPipeline
from uranus.ai.loaders import DataLoader_v1
from uranus.ai.models import Model_v1
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
model = Model_v1(dataset=dataset, n_hidden=32)
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
