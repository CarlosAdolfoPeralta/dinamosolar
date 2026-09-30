from math import cos, sin, trunc

import numpy as np
from _jit import njit


@njit(cache=True)
def _smooth_xy2(values, nx, ntit):
    """Aplica el filtro 2D usado para regularizar el término subgrid."""
    original = values.copy()
    smoothed = values.copy()
    factor = 0.5

    for i in range(1, nx - 1):
        for j in range(1, ntit - 1):
            smoothed[i, j] = (
                (1 - factor) ** 2 * original[i, j]
                + (1 - factor)
                * factor
                * (
                    original[i + 1, j]
                    + original[i - 1, j]
                    + original[i, j + 1]
                    + original[i, j - 1]
                )
                / 2
                + factor**2
                * (
                    original[i + 1, j + 1]
                    + original[i - 1, j - 1]
                    + original[i + 1, j - 1]
                    + original[i - 1, j + 1]
                )
                / 4
            )

    return smoothed


@njit(cache=True)
def Scero3(bphi, lamda, ur, utit, omega, br, btit, r, tit, nx, ntit, deltr, deltit):
    """Construye el aporte subgrid a la ecuación toroidal.

    Separa las contribuciones de cizalla rotacional y de advección meridional,
    y filtra el resultado antes de devolverlo.
    """
    dutitdtit = np.zeros((nx, ntit))
    durdtit = np.zeros((nx, ntit))
    durdr = np.zeros((nx, ntit))
    dutitdr = np.zeros((nx, ntit))
    d2urd2r = np.zeros((nx, ntit))
    d2utitd2t = np.zeros((nx, ntit))
    d2urd2rt = np.zeros((nx, ntit))
    d2utitd2rt = np.zeros((nx, ntit))

    dbtitdtit = np.zeros((nx, ntit))
    dbrdtit = np.zeros((nx, ntit))
    dbrdr = np.zeros((nx, ntit))
    dbtitdr = np.zeros((nx, ntit))

    dbphidr = np.zeros((nx, ntit))
    dbphidtit = np.zeros((nx, ntit))
    d2bphid2r = np.zeros((nx, ntit))
    d2bphid2t = np.zeros((nx, ntit))
    d2bphid2rt = np.zeros((nx, ntit))

    domegadtit = np.zeros((nx, ntit))
    domegadr = np.zeros((nx, ntit))
    d2omegad2r = np.zeros((nx, ntit))
    d2omegad2t = np.zeros((nx, ntit))
    d2omegad2rt = np.zeros((nx, ntit))

    fr = np.zeros((nx, ntit))
    ftit = np.zeros((nx, ntit))
    frtit = np.zeros((nx, ntit))
    ftitr = np.zeros((nx, ntit))
    ftittit = np.zeros((nx, ntit))
    gr = np.zeros((nx, ntit))
    gtit = np.zeros((nx, ntit))
    grtit = np.zeros((nx, ntit))
    grr = np.zeros((nx, ntit))
    gtittit = np.zeros((nx, ntit))

    sceroom = np.zeros((nx, ntit))
    scerou = np.zeros((nx, ntit))
    scero = np.zeros((nx, ntit))

    # Diferencias centradas de segundo orden para velocidad, campo y omega.
    for i in range(1, nx - 1):
        for j in range(1, ntit - 1):
            domegadtit[i, j] = (omega[i, j + 1] - omega[i, j - 1]) / (2 * deltit)
            domegadr[i, j] = (omega[i + 1, j] - omega[i - 1, j]) / (2 * deltr)
            d2omegad2r[i, j] = (
                omega[i + 1, j] + omega[i - 1, j] - 2 * omega[i, j]
            ) / deltr**2
            d2omegad2t[i, j] = (
                omega[i, j + 1] + omega[i, j - 1] - 2 * omega[i, j]
            ) / deltit**2
            d2omegad2rt[i, j] = (
                omega[i + 1, j + 1]
                + omega[i - 1, j - 1]
                - omega[i - 1, j + 1]
                - omega[i + 1, j - 1]
            ) / (4 * deltit * deltr)

            durdtit[i, j] = (ur[i, j + 1] - ur[i, j - 1]) / (2 * deltit)
            durdr[i, j] = (ur[i + 1, j] - ur[i - 1, j]) / (2 * deltr)
            d2urd2r[i, j] = (ur[i + 1, j] + ur[i - 1, j] - 2 * ur[i, j]) / deltr**2
            d2urd2rt[i, j] = (
                ur[i + 1, j + 1]
                + ur[i - 1, j - 1]
                - ur[i - 1, j + 1]
                - ur[i + 1, j - 1]
            ) / (4 * deltit * deltr)

            dutitdtit[i, j] = (utit[i, j + 1] - utit[i, j - 1]) / (2 * deltit)
            dutitdr[i, j] = (utit[i + 1, j] - utit[i - 1, j]) / (2 * deltr)
            d2utitd2t[i, j] = (
                utit[i, j + 1] + utit[i, j - 1] - 2 * utit[i, j]
            ) / deltit**2
            d2utitd2rt[i, j] = (
                utit[i + 1, j + 1]
                + utit[i - 1, j - 1]
                - utit[i - 1, j + 1]
                - utit[i + 1, j - 1]
            ) / (4 * deltit * deltr)

            dbtitdtit[i, j] = (btit[i, j + 1] - btit[i, j - 1]) / (2 * deltit)
            dbrdtit[i, j] = (br[i, j + 1] - br[i, j - 1]) / (2 * deltit)
            dbtitdr[i, j] = (btit[i + 1, j] - btit[i - 1, j]) / (2 * deltr)
            dbrdr[i, j] = (br[i + 1, j] - br[i - 1, j]) / (2 * deltr)

            dbphidtit[i, j] = (bphi[i, j + 1] - bphi[i, j - 1]) / (2 * deltit)
            dbphidr[i, j] = (bphi[i + 1, j] - bphi[i - 1, j]) / (2 * deltr)
            d2bphid2r[i, j] = (
                bphi[i + 1, j] + bphi[i - 1, j] - 2 * bphi[i, j]
            ) / deltr**2
            d2bphid2t[i, j] = (
                bphi[i, j + 1] + bphi[i, j - 1] - 2 * bphi[i, j]
            ) / deltit**2
            d2bphid2rt[i, j] = (
                bphi[i + 1, j + 1]
                + bphi[i - 1, j - 1]
                - bphi[i - 1, j + 1]
                - bphi[i + 1, j - 1]
            ) / (4 * deltit * deltr)

    for i in range(nx):
        for j in range(1, ntit - 1):
            aux1 = lamda**2 / (24 * r[i] ** 2)
            cosine = trunc(cos(tit[j]) * 1000) / 1000
            cosecant = trunc((1 / sin(tit[j])) * 1000) / 1000
            sine = trunc(sin(tit[j]) * 1000) / 1000
            cotangent = trunc((cos(tit[j]) / sin(tit[j])) * 1000) / 1000

            fr[i, j] = (
                3 * cosine * domegadtit[i, j]
                + sine
                * (
                    d2omegad2t[i, j]
                    + r[i] * domegadr[i, j]
                    - 2 * r[i] ** 2 * d2omegad2r[i, j]
                )
            )
            ftit[i, j] = (
                (3 + cos(2 * tit[j]))
                / 2
                * cosecant
                * domegadtit[i, j]
                - r[i] * sine * d2omegad2rt[i, j]
                - r[i] ** 2 * cosine * d2omegad2r[i, j]
            )
            frtit[i, j] = r[i] ** 2 * sine * d2omegad2rt[i, j]
            ftitr[i, j] = (
                r[i] * cosine * domegadr[i, j]
                + r[i] * sine * d2omegad2rt[i, j]
                - sine * domegadtit[i, j]
            )
            ftittit[i, j] = (
                cosine * domegadtit[i, j]
                + sine * d2omegad2t[i, j]
                - r[i] ** 2 * sine * d2omegad2r[i, j]
            )

            # Parte asociada al acoplamiento del campo con la rotación.
            sceroom[i, j] = aux1 * (
                btit[i, j] * ftit[i, j]
                + br[i, j] * fr[i, j]
                + dbtitdr[i, j] * frtit[i, j]
                + ftitr[i, j] * dbrdtit[i, j]
                + ftittit[i, j] * dbtitdtit[i, j]
            )

            gr[i, j] = (
                ur[i, j]
                + utit[i, j] * cotangent
                + r[i] * durdr[i, j]
                + r[i] ** 2 * d2urd2r[i, j]
                + r[i] * d2utitd2rt[i, j]
            ) * dbphidr[i, j]
            gtit[i, j] = (
                ur[i, j] * cotangent
                + utit[i, j] * cosecant**2
                + d2utitd2t[i, j]
                - r[i] * dutitdr[i, j]
                + r[i] * d2urd2rt[i, j]
            ) / r[i] * dbphidtit[i, j]
            grtit[i, j] = (
                -utit[i, j] + durdtit[i, j] + r[i] * dutitdr[i, j]
            ) * d2bphid2rt[i, j]
            grr[i, j] = r[i] ** 2 * durdr[i, j] * d2bphid2r[i, j]
            gtittit[i, j] = (
                ur[i, j] + dutitdtit[i, j]
            ) / r[i] * d2bphid2t[i, j]

            # Parte transportada por la circulación meridional.
            scerou[i, j] = -aux1 * (
                gr[i, j] + gtit[i, j] + grtit[i, j] + grr[i, j] + gtittit[i, j]
            )
            scero[i, j] = sceroom[i, j] + scerou[i, j]

    return _smooth_xy2(scero, nx, ntit)