import numpy as np


def guardaur(ur, m, ntit, ur1, ur2, ur3, ur4, ur5):
    """Guarda cinco perfiles radiales de la velocidad perturbada ur."""
    ur = np.asarray(ur)
    histories = (ur1, ur2, ur3, ur4, ur5)
    # Mantiene los mismos radios de muestreo que los historiales magnéticos.
    radial_indices = (21, 22, 29, 39, 49)

    for history, radial_index in zip(histories, radial_indices):
        history[m, :ntit] = ur[radial_index, :ntit]

    return histories