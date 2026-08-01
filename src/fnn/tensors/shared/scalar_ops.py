import math


def divide_scalar(left: float, right: float) -> float:
    if right == 0.0:
        if left == 0.0:
            return math.nan
        sign = math.copysign(1.0, left) * math.copysign(1.0, right)
        return math.copysign(math.inf, sign)
    return left / right


def log_scalar(value: float) -> float:
    """
    Treat log(0.0) as -inf because log values become more negative without
    limit as positive inputs get closer to zero.
    """
    if value == 0.0:
        return -math.inf
    if value < 0.0:
        return math.nan
    return math.log(value)


def sqrt_scalar(value: float) -> float:
    if value < 0.0:
        return math.nan
    return math.sqrt(value)


def sign_scalar(value: float) -> float:
    if value < 0.0:
        return -1.0
    if value > 0.0:
        return 1.0
    return 0.0
