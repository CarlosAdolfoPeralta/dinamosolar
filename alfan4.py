import numpy as np


def alfan4(
    lamda: float,
    r: np.ndarray,
    tit: np.ndarray,
    ur: np.ndarray,
    urtit: np.ndarray,
    omega: np.ndarray,
    nx: int,
    ntit: int,
    deltr: float,
    deltit: float,
    rtope: float,
    factalf: float,
) -> np.ndarray:
    """Calcula el coeficiente alfa cinemático a partir de gradientes de omega.

    Las fronteras quedan en cero, como en la rutina MATLAB original. Los
    argumentos ur, urtit y rtope se conservan por compatibilidad de interfaz.
    """
    r = np.asarray(r)
    tit = np.asarray(tit)
    omega = np.asarray(omega)

    alfa = np.zeros((nx, ntit), dtype=np.result_type(omega, r, float))

    # Compute all interior points at once; the outer rows and columns stay zero.
    domegadtit = (omega[1:-1, 2:] - omega[1:-1, :-2]) / (2 * deltit)
    domegadr = (omega[2:, 1:-1] - omega[:-2, 1:-1]) / (2 * deltr)

    # Proyección de los gradientes radial y angular de la rotación diferencial.
    aux2 = (
        np.sin(tit[1:-1])[None, :] * domegadtit
        - r[1:-1, None] * np.cos(tit[1:-1])[None, :] * domegadr
    )
    # Amplitud del efecto alfa fijada por lamda y factalf.
    alfa[1:-1, 1:-1] = (
        (lamda**2) / (24 * r[1:-1, None]) * factalf * aux2
    )

    return alfa