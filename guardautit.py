import numpy as np


def guardautit(utit, m, ntit, utit1, utit2, utit3, utit4, utit5):
    """Guarda cinco perfiles radiales de la velocidad perturbada u_theta."""
    utit = np.asarray(utit)
    histories = (utit1, utit2, utit3, utit4, utit5)
    # Usa las mismas capas que guardaur para comparar ambas componentes.
    radial_indices = (21, 22, 29, 39, 49)

    for history, radial_index in zip(histories, radial_indices):
        history[m, :ntit] = utit[radial_index, :ntit]

    return histories