SHELL := /bin/bash

# Load environment variables from activate.sh for use in Makefile
# This helps use variables like MLFLOW_PORT, etc. directly.
-include .env_vars

.PHONY: build jupyter mlflow-up mlflow-down clean
all:  build

# Helper to export vars from activate.sh into a temporary file for Makefile inclusion
.env_vars: activate.sh
	@grep "export " activate.sh | sed 's/export //' > .env_vars

build: .env_vars
	@bash activate.sh

jupyter: .env_vars
	@bash activate.sh && jupyter lab --IdentityProvider.token="" --ServerApp.password=""

mlflow-up: .env_vars
	@source activate.sh && \
	nohup mlflow server \
		--backend-store-uri sqlite:///$${MLFLOW_DB_PATH} \
		--default-artifact-root file://$${MLFLOW_ARTIFACT_PATH} \
		--host 0.0.0.0 \
		--port $${MLFLOW_PORT} > mlflow_server.log 2>&1 &
	@source activate.sh && echo "🚀 MLflow server starting in background (SQLite)... metrics @ http://localhost:$${MLFLOW_PORT}"

mlflow-down:
	@echo "🛑 Stopping MLflow server..."
	@pkill -f "mlflow server" || echo "No MLflow server running."

clean:
	@echo "🧹 Cleaning up..."
	rm -rf .uranus-env
	rm -rf .ai-env
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -f .env_vars
	rm -rf mlartifacts/
	rm -rf mlflow_db_data/
	rm -f mlflow.db
	rm -f mlflow_server.log
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "✨ Done!"