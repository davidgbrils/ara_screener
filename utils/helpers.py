"""Helper utility functions"""

import time
import functools
from typing import Callable, Any, Optional
from decimal import Decimal, ROUND_HALF_UP

def format_currency(value: float, currency: str = "IDR") -> str:
    """
    Format number as currency
    
    Args:
        value: Numeric value
        currency: Currency code
    
    Returns:
        Formatted currency string
    """
    if currency == "IDR":
        return f"Rp {value:,.0f}"
    return f"{currency} {value:,.2f}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format number as percentage
    
    Args:
        value: Numeric value (0-1 or 0-100)
        decimals: Decimal places
    
    Returns:
        Formatted percentage string
    """
    if abs(value) > 1:
        value = value / 100
    return f"{value * 100:.{decimals}f}%"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default on zero denominator
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
    
    Returns:
        Division result or default
    """
    if denominator == 0 or denominator is None:
        return default
    return numerator / denominator

def calculate_reward_risk(
    entry_price: float,
    stop_loss: float,
    take_profit: float
) -> dict:
    """
    Calculate reward-to-risk ratio
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
    
    Returns:
        Dictionary with reward, risk, and ratio
    """
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    ratio = safe_divide(reward, risk, 0.0)
    
    return {
        "reward": reward,
        "risk": risk,
        "ratio": ratio,
        "reward_pct": safe_divide(reward, entry_price, 0.0) * 100,
        "risk_pct": safe_divide(risk, entry_price, 0.0) * 100,
    }

def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying function calls on failure
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise
            
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator

def round_to_precision(value: float, precision: int = 2) -> float:
    """
    Round value to specified precision
    
    Args:
        value: Value to round
        precision: Decimal places
    
    Returns:
        Rounded value
    """
    decimal = Decimal(str(value))
    rounded = decimal.quantize(
        Decimal(10) ** -precision,
        rounding=ROUND_HALF_UP
    )
    return float(rounded)

def validate_ticker(ticker: str) -> bool:
    """
    Validate ticker format
    
    Args:
        ticker: Ticker symbol
    
    Returns:
        True if valid
    """
    if not ticker:
        return False
    # Basic validation: alphanumeric, max 10 chars
    return ticker.replace(".JK", "").isalnum() and len(ticker) <= 15

