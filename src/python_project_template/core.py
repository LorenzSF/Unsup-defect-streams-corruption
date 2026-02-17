from typing import List


def summarize(values: List[float]) -> dict:  # simple typed example
    """Return basic summary statistics for a list of numbers."""
    if not values:
        return {"count": 0, "mean": 0.0}
    total = sum(values)
    return {"count": len(values), "mean": total / len(values)}
