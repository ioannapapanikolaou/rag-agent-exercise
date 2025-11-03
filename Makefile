.PHONY: ingest run test eval
ingest:
	python -m app.ingest

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	PYTHONPATH=. pytest -q


eval:
	python -m eval.run_eval
