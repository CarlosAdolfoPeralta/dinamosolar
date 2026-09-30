from math import ceil

import numpy as np
from _jit import njit


@njit(cache=True)
def suavixy2d(var, nx, ntit):
    """Filtro de caja 3x3 para la función de corriente de la circulación."""
    var = np.asarray(var)
    aa = var.copy()
    midpoint = ceil(ntit / 2) - 1

    for i in range(1, nx - 1):
        for j in range(1, midpoint):
            var[i, j] = (
                aa[i, j]
                + aa[i + 1, j - 1]
                + aa[i + 1, j]
                + aa[i + 1, j + 1]
                + aa[i, j - 1]
                + aa[i, j]
                + aa[i, j + 1]
                + aa[i - 1, j - 1]
                + aa[i - 1, j]
                + aa[i - 1, j + 1]
            ) / 10
            # Fija el valor sobre el ecuador de la malla usada por este modelo.
            var[i, 15] = 0

    symmetry = np.sign(aa[29, 2]) * np.sign(aa[29, ntit - 3])
    for i in range(1, nx - 1):
        for j in range(midpoint + 1, ntit - 1):
            var[i, j] = var[i, ntit - j - 1] * symmetry

    return var