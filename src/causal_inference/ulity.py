

def normalize_postal_code(code: str) -> str:
    """Normalize postal code to 5-digit format (e.g., '94306-1234' -> '94306')"""
    return code.split("-")[0][:5]