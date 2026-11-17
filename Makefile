run_pipeline:
	python -m src.pipeline

run_app:
	python -m streamlit run app/app.py

.PHONY: run_pipeline run_app
