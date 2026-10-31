from src.feature_extraction import extract_entities, detect_risk_flags


def test_extract_entities():
    text = "Coupon of 7.5% for ISIN INE123A08012 payable on 15 December 2026 in INR."
    result = extract_entities(text)
    assert "INE123A08012" in result["isins"]
    assert "7.5%" in result["percentages"]
    assert "15 December 2026" in result["dates"]
    assert "INR" in result["currencies"]


def test_risk_flags():
    assert "DEFAULT" in detect_risk_flags("The issuer failed to pay interest and is in payment default.")
    assert detect_risk_flags("The issuer announced a regular coupon payment.") == []
