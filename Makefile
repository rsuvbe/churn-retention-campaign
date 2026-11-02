# Reproduce the whole project:  make setup && make all
PY := .venv/bin/python
NB := notebooks
NOTEBOOKS := 01_sql_eda_hypotheses 02_modeling 03_business_profit 04_ab_test_simulation 05_cross_sell_reactivation 06_deployment_monitoring

.PHONY: setup data notebooks slides app test all clean score

setup:            ## create venv and install pinned dependencies
	python3.12 -m venv .venv && $(PY) -m pip install -r requirements-lock.txt
	cd reports/deck && npm install

data:             ## build the DuckDB analytical database from the raw CSV
	$(PY) -c "from src.data import build_duckdb; print(build_duckdb())"

notebooks: data   ## execute all notebooks in order (outputs saved in place)
	@for nb in $(NOTEBOOKS); do \
		echo "== $$nb"; \
		$(PY) -m jupyter nbconvert --to notebook --execute --inplace \
			--ExecutePreprocessor.timeout=1200 $(NB)/$$nb.ipynb || exit 1; \
	done

slides:           ## build reports/churn_retention_presentation.pptx (needs Node.js)
	$(PY) -m src.slides

app:              ## launch the Streamlit scoring dashboard
	$(PY) -m streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -q tests

all: notebooks slides test   ## tests last, so the app smoke test sees the trained model

clean:
	rm -rf data/churn.duckdb models/* reports/figures/* reports/tables/*
	find data/processed -type f ! -name .gitkeep -delete

score:            ## batch-score a CSV:  make score IN=customers.csv OUT=scored.csv
	$(PY) -m src.score --input $(IN) --output $(OUT)
