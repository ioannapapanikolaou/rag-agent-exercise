from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


_PRICES_CACHE: Optional[Dict[str, List[Dict[str, float]]]] = None

# get the path to the prices json file
def _prices_path() -> Path:
    # repo_root / prices_stub / prices.json
    return Path(__file__).resolve().parents[2] / "prices_stub" / "prices.json"

# load the prices from the json file
def load_prices(force: bool = False) -> Dict[str, List[Dict[str, float]]]:
    global _PRICES_CACHE
    if _PRICES_CACHE is not None and not force:
        return _PRICES_CACHE
    with _prices_path().open("r", encoding="utf-8") as f:
        data = json.load(f)
    # normalize keys to uppercase
    _PRICES_CACHE = {k.upper(): v for k, v in data.items()}
    return _PRICES_CACHE

# get the price series for a given symbol
def _get_series(symbol: str) -> Optional[List[Dict[str, float]]]:
    prices = load_prices()
    return prices.get(symbol.upper())

# get the latest close price for a given symbol
def get_latest_close(symbol: str) -> Optional[Dict[str, float]]:
    series = _get_series(symbol)
    if not series:
        return None
    return series[-1]

# get the latest n close prices for a given symbol
def get_latest_n(symbol: str, n: int) -> Optional[List[Dict[str, float]]]:
    if n <= 0:
        return []
    series = _get_series(symbol)
    if not series:
        return None
    return series[-n:]

# calculate the percentage change between two prices
def pct_change(old: float, new: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old

# compare the performance of two symbols over a given number of points
def compare_performance(symbol_a: str, symbol_b: str, points: int = 10) -> Optional[Dict[str, float]]:
    a = get_latest_n(symbol_a, points)
    b = get_latest_n(symbol_b, points)
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return None
    a_ret = pct_change(a[0]["close"], a[-1]["close"])  # over available window
    b_ret = pct_change(b[0]["close"], b[-1]["close"])  # over available window
    rel = a_ret - b_ret
    return {"a_return": a_ret, "b_return": b_ret, "relative": rel}

# list all available symbols
def list_symbols() -> List[str]:
    return sorted(load_prices().keys())