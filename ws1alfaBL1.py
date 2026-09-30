import numpy as np
from _jit import njit

from suavixy2dbl import suavixy2dbl


@njit(cache=True)
def ws1alfaBL1(
    lamda,
    ur2,
    utit2,
    ur3,
    utit3,
    factalf,
    ws,
    delttime,
    omega,
    r,
    tit,
    nx,
    ntit,
    deltr,
    deltit,
):
    """Actualiza la vorticidad de realimentación y la contribución alfa BL."""
    ws0 = np.zeros((nx, ntit))
    ws1 = np.asarray(ws).copy()
    alf_bl = np.zeros((nx, ntit))
    k1 = np.zeros((nx, ntit))
    k2 = np.zeros((nx, ntit))
    k3 = np.zeros((nx, ntit))
    k4 = np.zeros((nx, ntit))
    q1 = np.zeros((nx, ntit))
    q2 = np.zeros((nx, ntit))
    q3 = np.zeros((nx, ntit))
    dws1dtit = np.zeros((nx, ntit))
    dws1dr = np.zeros((nx, ntit))
    dws0dtit = np.zeros((nx, ntit))
    dws0dr = np.zeros((nx, ntit))
    dur0dtit = np.zeros((nx, ntit))
    dur0dr = np.zeros((nx, ntit))
    dutit0dtit = np.zeros((nx, ntit))
    dutit0dr = np.zeros((nx, ntit))
    dur1dtit = np.zeros((nx, ntit))
    dur1dr = np.zeros((nx, ntit))
    dutit1dtit = np.zeros((nx, ntit))
    dutit1dr = np.zeros((nx, ntit))

    ur0 = np.asarray(ur2)
    utit0 = np.asarray(utit2)
    ur1 = np.asarray(ur3)
    utit1 = np.asarray(utit3)
    omega = np.asarray(omega)

    # ws0 representa la fuente ligada a los gradientes de rotación omega.
    for i in range(1, nx - 1):
        for j in range(1, ntit - 1):
            sine = np.sin(tit[j])
            cosine = np.cos(tit[j])
            ws0[i, j] = sine * (
                sine * (omega[i, j + 1] - omega[i, j - 1]) / (2 * deltit)
                - r[i] * cosine * (omega[i + 1, j] - omega[i - 1, j]) / (2 * deltr)
            )

    for i in range(1, nx - 1):
        for j in range(1, ntit - 1):
            dws1dtit[i, j] = (ws1[i, j + 1] - ws1[i, j - 1]) / (2 * deltit)
            dws1dr[i, j] = (ws1[i + 1, j] - ws1[i - 1, j]) / (2 * deltr)
            dws0dtit[i, j] = (ws0[i, j + 1] - ws0[i, j - 1]) / (2 * deltit)
            dws0dr[i, j] = (ws0[i + 1, j] - ws0[i - 1, j]) / (2 * deltr)

            dur1dtit[i, j] = (ur1[i, j + 1] - ur1[i, j - 1]) / (2 * deltit)
            dur1dr[i, j] = (ur1[i + 1, j] - ur1[i - 1, j]) / (2 * deltr)
            dur0dtit[i, j] = (ur0[i, j + 1] - ur0[i, j - 1]) / (2 * deltit)
            dur0dr[i, j] = (ur0[i + 1, j] - ur0[i - 1, j]) / (2 * deltr)

            dutit1dtit[i, j] = (utit1[i, j + 1] - utit1[i, j - 1]) / (2 * deltit)
            dutit1dr[i, j] = (utit1[i + 1, j] - utit1[i - 1, j]) / (2 * deltr)
            dutit0dtit[i, j] = (utit0[i, j + 1] - utit0[i, j - 1]) / (2 * deltit)
            dutit0dr[i, j] = (utit0[i + 1, j] - utit0[i - 1, j]) / (2 * deltr)

    for i in range(1, nx - 1):
        for j in range(1, ntit - 1):
            sine = np.sin(tit[j])
            cosine = np.cos(tit[j])
            radius = r[i]

            k1[i, j] = 2 * omega[i, j] * (
                cosine * sine * dur1dr[i, j]
                + cosine**2 * dutit1dr[i, j]
                - sine**2 / radius * dur1dtit[i, j]
                - ur1[i, j] / radius * sine * cosine
                + utit1[i, j] / radius * sine**2
                - sine / radius * cosine * dutit1dtit[i, j]
            )
            k2[i, j] = ws0[i, j] * (
                cosine**2 * dur1dr[i, j]
                - cosine * sine * dutit1dr[i, j]
                - sine / radius * cosine * dur1dtit[i, j]
                + sine**2 / radius * ur1[i, j]
                + sine * cosine / radius * utit1[i, j]
                + sine**2 / radius * dutit1dtit[i, j]
            )
            k3[i, j] = (ur1[i, j] * sine + utit1[i, j] * cosine) * (
                ws0[i, j] / (radius * sine)
                + sine * dws0dr[i, j]
                + cosine / radius * dws0dtit[i, j]
            )
            k4[i, j] = (ur1[i, j] * cosine - utit1[i, j] * sine) * (
                cosine * dws0dr[i, j] - sine / radius * dws0dtit[i, j]
            )

            q1[i, j] = (ur0[i, j] * sine + utit0[i, j] * cosine) * (
                sine * dws1dr[i, j]
                + cosine / radius * dws1dtit[i, j]
                + ws1[i, j] / (radius * sine)
            )
            q2[i, j] = (ur0[i, j] * cosine - utit0[i, j] * sine) * (
                cosine * dws1dr[i, j] - sine / radius * dws1dtit[i, j]
            )
            q3[i, j] = ws1[i, j] * (
                cosine**2 * dur0dr[i, j]
                - cosine * sine * dutit0dr[i, j]
                - sine / radius * cosine * dur0dtit[i, j]
                + sine**2 / radius * ur0[i, j]
                + sine * cosine / radius * utit0[i, j]
                + sine**2 / radius * dutit0dtit[i, j]
            )

    # k reúne términos de la perturbación y q su interacción con el flujo base.
    k = k1 - k2 - k3 - k4
    q = -q1 - q2 - q3
    ws1 = (k - q) * delttime + ws1

    # La vorticidad actualizada alimenta el alpha no local del modelo.
    for i in range(1, nx - 1):
        for j in range(1, ntit - 1):
            alf_bl[i, j] = lamda**2 / (24 * r[i] * np.sin(tit[j])) * factalf * ws1[i, j]

    ws = suavixy2dbl(ws1, nx, ntit)
    alf_bl = suavixy2dbl(alf_bl, nx, ntit)
    return ws, alf_bl