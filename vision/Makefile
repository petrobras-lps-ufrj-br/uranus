# Makefile for Cognite Open Data Server

PYTHON = python3
VENV = venv_vision
BIN = $(VENV)/bin

.PHONY: help build up down restart logs ps clean shell venv infra run-local

help:
	@echo "Available commands:"
	@echo "  make venv      - Create virtualenv and install dependencies"
	@echo "  make infra     - Start infrastructure (DB, pgAdmin, Grafana) in Docker"
	@echo "  make run-local - Run the app locally using virtualenv"
	@echo "  make build     - Build the docker images"
	@echo "  make up        - Start all services (including app) in Docker"
	@echo "  make down      - Stop and remove the containers"
	@echo "  make restart   - Restart the services"
	@echo "  make logs      - View the logs from all services"
	@echo "  make ps        - List running containers"
	@echo "  make clean     - Remove containers and volumes (Warning: Deletes data)"

build:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

infra:
	docker compose up -d db pgadmin grafana

run:
	$(BIN)/python main.py


up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

ps:
	docker compose ps

clean:
	docker compose down -v
	rm -rf $(VENV)

shell:
	docker compose exec app /bin/bash
