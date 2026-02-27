
export VIRTUALENV_NAMESPACE='.uranus-env'
export LOGURU_LEVEL="DEBUG"
export VIRTUALENV_PATH=$PWD/$VIRTUALENV_NAMESPACE

export URANUS_DATA_PATH=$PWD/data
export MLFLOW_PORT=8000
export MLFLOW_ARTIFACT_PATH="$PWD/mlartifacts"
export MLFLOW_DB_PATH="$PWD/mlflow.db"
export MLFLOW_TRACKING_URI="http://localhost:$MLFLOW_PORT"

if [ -d "$VIRTUALENV_PATH" ]; then
    echo "$VIRTUALENV_PATH exists."
    source $VIRTUALENV_PATH/bin/activate
else
    virtualenv -p python ${VIRTUALENV_PATH}
    source $VIRTUALENV_PATH/bin/activate
    pip install -e .
fi
