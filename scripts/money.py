"""Exact paise aggregation and two-decimal serialization of monetary totals."""

from decimal import Decimal, InvalidOperation


def paise(value):
    """Reject invalid/sub-paise inputs; do not silently round source amounts."""
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Money must be a finite nonnegative amount in whole paise") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("Money must be a finite nonnegative amount in whole paise")
    units = amount * 100
    if units != units.to_integral_value():
        raise ValueError("Money must be a finite nonnegative amount in whole paise")
    return int(units)


def customer_spend(orders):
    # Python integers avoid overflow and summation-order dependence. Convert to
    # float only once, at the model boundary, after the exact total is known.
    units = orders["order_amount"].map(paise).astype(object)
    totals = units.groupby(orders["customer_id"], sort=True).sum()
    return totals.map(lambda value: float(Decimal(int(value)) / 100))


def write_money_csv(frame, path, columns, *, index=False):
    serialized = frame.copy()
    for column in columns:
        serialized[column] = serialized[column].map(
            lambda value: format(Decimal(paise(value)) / 100, ".2f")
        )
    serialized.to_csv(path, index=index, lineterminator="\n")
