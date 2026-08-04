.PHONY: run migrate check-hf docker-up docker-down reset-chroma

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

migrate:
	alembic upgrade head

check-hf:
	python scripts/check_hf_models.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

reset-chroma:
	python scripts/reset_chroma.py