from math import cos, sin

import numpy as np
from _jit import njit


@njit(cache=True)
def velmeri4(rtope, rcero, rp, nx, ntit, r, tit, sigma, ro):
    """Define el flujo meridional de fondo mediante perfiles analíticos."""
    ur = np.zeros((nx, ntit))
    utit = np.zeros((nx, ntit))

    # El flujo impuesto se restringe a r > rp y depende de la densidad local.
    for i in range(nx):
        if r[i] > rp:
            for j in range(ntit):
                cosine = cos(tit[j])
                sine = sin(tit[j])

                ur[i, j] = (
                    sigma
                    / (r[i] * ro[i])
                    * (rtope - r[i])
                    * (r[i] - rp)
                    * (3 * cosine**2 - 1)
                )
                utit[i, j] = (
                    (sigma / ro[i])
                    * sine**3
                    * cosine
                    * (r[i] - (rtope + rp) / 2)
                )

    return ur, utit