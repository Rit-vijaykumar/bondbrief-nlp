"""Lightweight, explainable extraction helpers for financial announcements."""
import re
from typing import Dict, List

ISIN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}[0-9]\b")
DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(r"\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b", re.I),
]
AMOUNT_RE = re.compile(r"(?:INR|USD|EUR|GBP|₹|\$|€|£)\s?\d[\d,]*(?:\.\d+)?", re.I)
PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?%")
RISK_TERMS = {
    "DEFAULT": ["default", "failed to pay", "payment default", "missed payment"],
    "RESTRUCTURING": ["restructuring", "debt rescheduling", "restructuring plan"],
    "CREDIT_DOWNGRADE": ["downgrade", "rating revised downward", "rating lowered"],
    "PAYMENT_DELAY": ["payment delayed", "延期", "postponed payment", "deferred payment"],
}


def extract_entities(text: str) -> Dict[str, List[str]]:
    text = text or ""
    dates = []
    for pattern in DATE_PATTERNS:
        dates.extend(pattern.findall(text))
    currencies = sorted(set(re.findall(r"\b(?:INR|USD|EUR|GBP|JPY|AUD|CAD)\b|[₹$€£]", text, re.I)))
    return {
        "isins": sorted(set(ISIN_RE.findall(text.upper()))),
        "dates": sorted(set(dates)),
        "amounts": sorted(set(AMOUNT_RE.findall(text))),
        "percentages": sorted(set(PERCENT_RE.findall(text))),
        "currencies": currencies,
    }


def detect_risk_flags(text: str) -> List[str]:
    lowered = (text or "").lower()
    return [flag for flag, terms in RISK_TERMS.items() if any(term in lowered for term in terms)]
