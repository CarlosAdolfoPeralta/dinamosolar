import numpy as np
from _jit import njit

from suavixy import suavixy
from suavixy2d import suavixy2d


@njit(cache=True)
def leap2(r, tit, nx, ntit, rtope, mu0, omega0, b, br, btit, deltr, deltit, ro):
    """Calcula la circulación meridional inducida por la fuerza magnética.

    Primero resuelve una función de corriente discreta y luego obtiene las
    velocidades de realimentación a partir de sus derivadas espaciales.
    """
    r = np.asarray(r)
    tit = np.asarray(tit)
    omega0 = np.asarray(omega0)
    b = np.asarray(b)
    br = np.asarray(br)
    btit = np.asarray(btit)
    ro = np.asarray(ro)

    fi = np.zeros((nx, ntit))
    ur = np.zeros((nx, ntit))
    utit = np.zeros((nx, ntit))
    ur2 = np.zeros((nx, ntit))
    utit2 = np.zeros((nx, ntit))
    a = np.zeros((nx, ntit))
    b_coeff = np.zeros((nx, ntit))
    c = np.zeros((nx, ntit))

    # Fuerza de Lorentz proyectada en la ecuación elíptica para la función fi.
    for j in range(14, 0, -1):
        for i in range(1, nx - 1):
            cte = 2 * mu0 * omega0[4, 4]
            dbdr = (b[i + 1, j] - b[i - 1, j]) / (2 * deltr)
            dbdtit = (b[i, j + 1] - b[i, j - 1]) / (2 * deltit)

            c[i, j] = (r[i] / cte) * (
                btit[i, j] * (b[i, j] * np.cos(tit[j]) + np.sin(tit[j]) * dbdtit)
                + br[i, j]
                * (b[i, j] * np.sin(tit[j]) + r[i] * np.sin(tit[j]) * dbdr)
            )
            a[i, j] = -r[i] * np.cos(tit[j])
            b_coeff[i, j] = np.sin(tit[j])

    for j in range(14, 0, -1):
        for i in range(1, nx - 1):
            fi[i, j] = (
                a[i, j] * fi[i - 1, j] / deltr
                + b_coeff[i, j] * fi[i, j - 1] / deltit
                + c[i, j]
            ) / (a[i, j] / deltr + b_coeff[i, j] / deltit)

    # Extiende el campo por simetría respecto del ecuador solar.
    fi[:, 15] = 0
    for i in range(nx):
        for j in range(16, ntit - 1):
            fi[i, j] = -fi[i, ntit - j - 1]

    fi = suavixy2d(fi, nx, ntit)

    # Las derivadas de fi se convierten en ur y u_theta; ro escala la respuesta.
    for i in range(1, nx - 1):
        for j in range(1, ntit - 1):
            dfidr = (fi[i + 1, j] - fi[i - 1, j]) / (2 * deltr)
            dfidtit = (fi[i, j + 1] - fi[i, j - 1]) / (2 * deltit)
            ur[i, j] = (1 / r[i]) * (
                np.cos(tit[j]) / np.sin(tit[j]) * fi[i, j] + dfidtit
            )
            utit[i, j] = -(1 / r[i]) * (fi[i, j] + r[i] * dfidr)
            ur2[i, j] = 1 / (r[i] ** 2 * np.sin(tit[j]) * ro[i]) * dfidtit
            utit2[i, j] = -1 / (r[i] * np.sin(tit[j]) * ro[i]) * dfidr

    ur2 = suavixy(ur2, nx, ntit)
    utit2 = suavixy(utit2, nx, ntit)
    return ur2, utit2