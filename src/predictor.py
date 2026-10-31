"""Load the trained classifier and return a structured analysis."""
from pathlib import Path
import joblib
from src.feature_extraction import extract_entities, detect_risk_flags

MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "announcement_classifier.joblib"


def analyze_announcement(text: str, model_path: Path = MODEL_PATH):
    if not text or not text.strip():
        raise ValueError("Please provide announcement text.")
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run `python -m src.train_model` first.")
    model = joblib.load(model_path)
    probabilities = model.predict_proba([text])[0]
    best_idx = probabilities.argmax()
    return {
        "category": str(model.classes_[best_idx]),
        "confidence": float(probabilities[best_idx]),
        "entities": extract_entities(text),
        "risk_flags": detect_risk_flags(text),
        "probabilities": {str(label): float(prob) for label, prob in zip(model.classes_, probabilities)},
    }
