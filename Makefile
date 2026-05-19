.PHONY: demo backend frontend install install-backend install-frontend clean

# One-command demo: install deps, start backend + frontend
demo: install
	@echo "Starting AlignAI demo..."
	@make -j2 _backend_run _frontend_run

_backend_run:
	cd backend && .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

_frontend_run:
	cd frontend && npm run dev

# Install all dependencies
install: install-backend install-frontend

install-backend:
	@echo "Setting up Python backend..."
	cd backend && python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -e ".[dev]"
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from .env.example — edit it to add your OPENROUTER_API_KEY"; fi
	@mkdir -p data/uploads data/artifacts data/results

install-frontend:
	@echo "Installing frontend dependencies..."
	cd frontend && npm install

# Run backend only
backend: install-backend
	cd backend && .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Run frontend only
frontend: install-frontend
	cd frontend && npm run dev

# Run backend tests
test-backend:
	cd backend && .venv/bin/pytest tests/ -v

# Run frontend type-check
typecheck-frontend:
	cd frontend && npx tsc --noEmit

# Generate synthetic test data
generate-test-data:
	cd backend && .venv/bin/python scripts/generate_synthetic_test_data.py

# Clean build artifacts
clean:
	rm -rf backend/.venv backend/__pycache__ backend/**/__pycache__
	rm -rf frontend/node_modules frontend/dist
	rm -rf data/uploads/* data/artifacts/* data/results/* data/jobs.sqlite

# Docker
docker-build:
	docker compose build

docker-up:
	docker compose up

docker-down:
	docker compose down
