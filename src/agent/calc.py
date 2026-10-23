"""Calculator logic (kept free of LangChain so it is easy to test)."""

OPERATIONS = ("add", "subtract", "multiply", "divide", "power")
MAX_EXPONENT = 1000


def compute(a: float, b: float, operation: str) -> float:
    """Apply ``operation`` to ``a`` and ``b``. Raises ValueError on bad input."""
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b
    if operation == "power":
        if abs(b) > MAX_EXPONENT:
            raise ValueError(f"Exponent too large (limit is {MAX_EXPONENT}).")
        try:
            result = a**b
        except (OverflowError, ZeroDivisionError) as exc:
            raise ValueError(f"Could not compute power: {exc}") from exc
        if isinstance(result, complex):
            raise ValueError("Result is not a real number.")
        return result
    raise ValueError(
        f"Unknown operation '{operation}'. Use one of: {', '.join(OPERATIONS)}."
    )


def format_number(value: float) -> str:
    """Show whole numbers without a trailing .0 and trim long decimals."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.10g}"
