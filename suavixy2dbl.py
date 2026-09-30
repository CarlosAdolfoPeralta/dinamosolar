from math import ceil

import numpy as np
from _jit import njit


@njit(cache=True)
def suavixy2dbl(var, nx, ntit):
    """Filtro 2D con mezcla centro-vecinos y simetría ecuatorial."""
    var = np.asarray(var)
    a1 = var.copy()
    factor = 0.45
    midpoint = ceil(ntit / 2) - 1
    symmetry = np.sign(a1[29, 2]) * np.sign(a1[29, ntit - 3])

    for i in range(1, nx - 1):
        for j in range(1, midpoint):
            var[i, j] = (
                (1 - factor) ** 2 * a1[i, j]
                + (1 - factor)
                * factor
                * (a1[i + 1, j] + a1[i - 1, j] + a1[i, j + 1] + a1[i, j - 1])
                / 2
                + factor**2
                * (
                    a1[i + 1, j + 1]
                    + a1[i - 1, j - 1]
                    + a1[i + 1, j - 1]
                    + a1[i - 1, j + 1]
                )
                / 4
            )
            var[i, 15] = 0

    for i in range(1, nx - 1):
        for j in range(midpoint + 1, ntit - 1):
            var[i, j] = var[i, ntit - j - 1] * symmetry

    return var