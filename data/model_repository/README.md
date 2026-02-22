# 🗄 Triton Model Repository Structure

Triton Inference Server requires a specific directory structure to discover and load models. Below is the proposed schema for **Uranus**:

```text
data/model_repository/
├── <model_name>/
│   ├── config.pbtxt        # Model configuration (backend, inputs, outputs)
│   ├── 1/                  # Version number (must be numeric)
│   │   └── model.pt        # Model file (e.g., PyTorch, ONNX, etc.)
│   └── 2/                  # (Optional) Newer version
│       └── model.pt
└── sample_model/
    ├── config.pbtxt
    └── 1/
        └── model.pt
```

### 📋 Configuration (`config.pbtxt`)

Example for a PyTorch (LibTorch) model:

```protobuf
name: "sample_model"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }
]
```

### 🚀 Backend Support

*   **PyTorch**: `pytorch_libtorch` (uses `model.pt`)
*   **ONNX**: `onnxruntime_onnx` (uses `model.onnx`)
*   **TensorFlow**: `tensorflow_savedmodel` (uses `saved_model/`)
*   **Python**: `python` (uses `model.py`)
