from math import ceil

import numpy as np
from _jit import njit


@njit(cache=True)
def etauv4(nx, ntit, ur, utit, r, tit, cs, deltr, deltit, omega, br, btit, b, ro):
    """Estima eta turbulenta a partir de cizalla de velocidad y campo magnético.

    El factor cs fija la escala de mezcla; la raíz combina invariantes locales
    de deformación, cizalla diferencial y gradientes magnéticos.
    """
    etauv = np.zeros((nx, ntit))
    dutitdr = np.zeros((nx, ntit))
    dutitdtit = np.zeros((nx, ntit))
    durdr = np.zeros((nx, ntit))
    durdtit = np.zeros((nx, ntit))
    domegadr = np.zeros((nx, ntit))
    domegadtit = np.zeros((nx, ntit))

    mu0 = 4 * np.pi * 10 ** (-7)
    b2 = np.zeros((nx, ntit))
    br2 = np.zeros((nx, ntit))
    btit2 = np.zeros((nx, ntit))
    for i in range(nx):
        for j in range(ntit):
            b2[i, j] = b[i, j] / np.sqrt(mu0 * ro[i])
            br2[i, j] = b[i, j] / np.sqrt(mu0 * ro[i])
            btit2[i, j] = b[i, j] / np.sqrt(mu0 * ro[i])

    dbdtit = np.zeros((nx, ntit))
    dbdr = np.zeros((nx, ntit))
    dbrdtit = np.zeros((nx, ntit))
    dbtitdr = np.zeros((nx, ntit))
    for i in range(1, nx - 1):
        for j in range(1, ntit - 1):
            dbdtit[i, j] = (b2[i, j + 1] - b2[i, j - 1]) / (2 * deltit)
            dbdr[i, j] = (b2[i + 1, j] - b2[i - 1, j]) / (2 * deltr)
            dbrdtit[i, j] = (br2[i, j + 1] - br2[i, j - 1]) / (2 * deltit)
            dbtitdr[i, j] = (btit2[i + 1, j] - btit2[i - 1, j]) / (2 * deltr)

    val2 = np.zeros((nx, ntit))
    for i in range(1, nx - 1):
        for j in range(1, ntit - 1):
            val2[i, j] = (
                (
                    (1 / (r[i] * np.sin(tit[j])))
                    * (np.cos(tit[j]) * b[i, j] + np.sin(tit[j]) * dbdtit[i, j])
                ) ** 2
                + (1 / r[i] * (b[i, j] + r[i] * dbdr[i, j])) ** 2
                + (1 / r[i] * (btit[i, j] + r[i] * dbtitdr[i, j] - dbrdtit[i, j])) ** 2
            ) * 5 / 7

    # A cada lado del ecuador se usan diferencias angulares orientadas hacia él.
    midpoint = ceil(ntit / 2) - 1
    for i in range(1, nx - 1):
        for j in range(midpoint, 0, -1):
            durdr[i, j] = (ur[i + 1, j] - ur[i - 1, j]) / (2 * deltr)
            durdtit[i, j] = (ur[i, j] - ur[i, j - 1]) / (r[i] * 2 * deltit)

            dutitdr[i, j] = (utit[i + 1, j] - utit[i - 1, j]) / (2 * deltr)
            dutitdtit[i, j] = (utit[i, j] - utit[i, j - 1]) / (r[i] * 2 * deltit)

            domegadr[i, j] = (omega[i + 1, j] - omega[i, j]) / deltr
            domegadtit[i, j] = (omega[i, j] - omega[i, j - 1]) / deltit

        for j in range(midpoint, ntit - 1):
            durdr[i, j] = (ur[i + 1, j] - ur[i - 1, j]) / (2 * deltr)
            durdtit[i, j] = (ur[i, j] - ur[i, j + 1]) / (r[i] * 2 * deltit)

            dutitdr[i, j] = (utit[i + 1, j] - utit[i - 1, j]) / (2 * deltr)
            dutitdtit[i, j] = (utit[i, j] - utit[i, j + 1]) / (r[i] * 2 * deltit)

            domegadr[i, j] = (omega[i + 1, j] - omega[i, j]) / deltr
            domegadtit[i, j] = (omega[i, j] - omega[i, j + 1]) / deltit

        for j in range(1, ntit - 1):
            value = (
                2 * durdr[i, j] ** 2
                + 2 * ((1 / r[i]) * dutitdtit[i, j] + ur[i, j] / r[i]) ** 2
                + (durdtit[i, j] / r[i] + dutitdr[i, j] - utit[i, j] / r[i]) ** 2
                + 2
                * (
                    ur[i, j] / r[i]
                    + utit[i, j] * np.cos(tit[j]) / np.sin(tit[j]) / r[i]
                ) ** 2
                + (np.sin(tit[j]) * domegadtit[i, j]) ** 2
                + (r[i] * np.sin(tit[j]) * domegadr[i, j]) ** 2
            )
            etauv[i, j] = (cs**2) * deltr * r[i] * deltit * np.sqrt(value + val2[i, j])

    return etauv