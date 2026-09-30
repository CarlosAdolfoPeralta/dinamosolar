import numpy as np
from _jit import njit


@njit(cache=True)
def laplaciano(atercer, lamda, nx, ntit, deltr, deltit, r, tit):
    """Aplica el operador laplaciano esférico al término de tercer orden."""
    atercer = np.asarray(atercer)
    sosuav = np.zeros((nx, ntit))

    # Se dejan tres celdas de borde sin filtrar para evitar stencils incompletos.
    for j in range(3, ntit - 3):
        for i in range(3, nx - 3):
            d2sodr2 = (atercer[i + 1, j] - 2 * atercer[i, j] + atercer[i - 1, j]) / deltr**2
            dsodr = (atercer[i + 1, j] - atercer[i - 1, j]) / (2 * deltr)
            dsodtit = (atercer[i, j + 1] - atercer[i, j - 1]) / (2 * deltit)
            d2sodtit2 = (
                atercer[i, j + 1] - 2 * atercer[i, j] + atercer[i, j - 1]
            ) / deltit**2

            sosuav[i, j] = (
                d2sodr2
                + 2 / r[i] * dsodr
                + 1 / r[i] ** 2 * d2sodtit2
                + np.cos(tit[j]) / (np.sin(tit[j]) * r[i] ** 2) * dsodtit
            )

    atercer[:, :] = atercer[:, :] + sosuav * lamda**2 / 24 * 10**-1
    return atercer