SHELL := /bin/bash

.PHONY: build jupyter clean
all:  build

build:
	@bash activate.sh

jupyter:
	@bash activate.sh && jupyter lab --IdentityProvider.token="" --ServerApp.password=""

clean:
	@echo "🧹 Cleaning up..."
	rm -rf .uranus-env
	rm -rf .ai-env
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "✨ Done!"