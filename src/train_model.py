"""Train and evaluate the starter announcement classifier."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "announcements.csv"
ARTIFACTS = ROOT / "artifacts"


def main():
    df = pd.read_csv(DATA).dropna(subset=["text", "label"])
    X_train, X_test, y_train, y_test = train_test_split(
        df.text, df.label, test_size=0.25, random_state=42, stratify=df.label
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, strip_accents="unicode", sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / "announcement_classifier.joblib")
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "train_rows": len(X_train), "test_rows": len(X_test),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Saved model to {ARTIFACTS / 'announcement_classifier.joblib'}")


if __name__ == "__main__":
    main()
