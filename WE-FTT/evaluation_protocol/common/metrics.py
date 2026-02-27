from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int


def confusion_from_binary(y_true, y_pred) -> Confusion:
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        t1 = int(t) == 1
        p1 = int(p) == 1
        if t1 and p1:
            tp += 1
        elif (not t1) and p1:
            fp += 1
        elif (not t1) and (not p1):
            tn += 1
        else:
            fn += 1
    return Confusion(tp=tp, fp=fp, tn=tn, fn=fn)


def matthews_corrcoef(c: Confusion) -> float:
    num = c.tp * c.tn - c.fp * c.fn
    den = (c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn)
    if den <= 0:
        return 0.0
    return num / math.sqrt(den)


def fpr(c: Confusion) -> float:
    denom = c.fp + c.tn
    return (c.fp / denom) if denom > 0 else 0.0


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # 95%
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    adj = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    lower = (centre - adj) / denom
    upper = (centre + adj) / denom
    return (max(0.0, lower), min(1.0, upper))

