# utils.py
from dateutil import parser
import re

def parse_date(datestr):
    # robust ISO-ish parser
    return parser.parse(datestr).date()

def normalize_unit_price(value, unit):
    """
    value: numeric, unit: string (e.g., 'Rs/Quintal', 'Rs/Qtl', 'Rs/Kg', 'Rs/kg', 'Rs/Quintal (processed)')
    returns price per kg (float)
    """
    if value is None:
        return None
    v = float(value)
    u = (unit or "").lower()
    if "quint" in u or "qtl" in u or "per q" in u:
        # per quintal => divide by 100 to get per kg
        return v / 100.0
    if "kg" in u:
        return v
    # fallback: assume per quintal
    return v / 100.0
