# BondBrief NLP – Financial Announcement Intelligence

BondBrief NLP is a portfolio project that classifies fixed-income announcements and extracts useful entities and risk signals from announcement text.

## Features
- Classifies announcements into coupon payment, full/partial redemption, tender offer, bond call, consent solicitation, default, or other.
- Extracts ISINs, dates, currencies, amounts, and percentages.
- Flags risk phrases such as default, restructuring, credit downgrade, and payment delay.
- Includes a Streamlit interface for single announcements and CSV batch analysis.
- Includes a starter dataset, model training script, and unit tests.

## Tech stack
Python, pandas, scikit-learn, TF-IDF, Logistic Regression, Streamlit, pytest.

## Setup
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train_model
streamlit run app/streamlit_app.py
```

Run tests with `python -m pytest -q`.

## Dataset and limitations
The included announcement examples are synthetic starter data created for demonstration and learning. Model metrics from this small dataset are not evidence of production readiness. For real-world use, obtain properly licensed, reviewed announcement data; expand the dataset; validate labels; and evaluate by issuer/time period to reduce leakage. Predictions should be reviewed by an analyst and must not be treated as investment or operational instructions.

## Project structure
- `src/feature_extraction.py` – entity and risk phrase extraction
- `src/predictor.py` – model loading and prediction pipeline
- `src/train_model.py` – training and evaluation
- `app/streamlit_app.py` – interactive dashboard
- `data/announcements.csv` – synthetic starter dataset
- `tests/` – unit tests
