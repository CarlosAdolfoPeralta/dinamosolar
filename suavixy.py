from math import ceil

import numpy as np
from _jit import njit


@njit(cache=True)
def suavixy(var, nx, ntit):
    """Filtra el campo y reconstruye el hemisferio opuesto con su paridad."""
    var = np.asarray(var)
    a1 = var.copy()
    factor = 0.27
    midpoint = ceil(ntit / 2) - 1
    # El producto de signos detecta simetría par (+1) o impar (-1) ecuatorial.
    symmetry = np.sign(a1[29, 2]) * np.sign(a1[29, ntit - 3])

    if symmetry == -1:
        var[:, midpoint] = 0
    else:
        for i in range(nx):
            var[i, midpoint] = (var[i, midpoint + 1] + var[i, midpoint - 1]) / 2

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

    for i in range(1, nx - 1):
        for j in range(midpoint + 1, ntit - 1):
            var[i, j] = var[i, ntit - j - 1] * symmetry

    if symmetry == -1:
        var[:, midpoint] = 0
    else:
        i = nx - 2
        var[i, midpoint] = (var[i, midpoint + 1] + var[i, midpoint - 1]) / 2

    return var