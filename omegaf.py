from math import cos, erf, pi

import numpy as np
from _jit import njit


@njit(cache=True)
def omegaf(r, rtope, nx, tit, ntit, n):
    """Construye omega con rotación diferencial superficial y núcleo uniforme."""
    alfa2 = -62.69 * pi * 2e-9
    alfa4 = -67.15 * pi * 2e-9

    omega_eq = n * 460.7 * pi * 2e-9
    omega_rz = n * 432.8 * pi * 2e-9
    # La transición radial suaviza el empalme entre zona radiativa y convectiva.
    dt = 0.05 * rtope
    rt = 0.7 * rtope

    omega = np.zeros((nx, ntit))

    for j in range(ntit):
        cos_tit = cos(tit[j])
        omega_scz = omega_eq + alfa2 * cos_tit**2 + alfa4 * cos_tit**4

        for i in range(nx):
            transition = 0.5 * (1 + erf(2 * ((r[i] - rt) / dt)))
            omega[i, j] = omega_rz + transition * (omega_scz - omega_rz)

    omega *= n
    return omega