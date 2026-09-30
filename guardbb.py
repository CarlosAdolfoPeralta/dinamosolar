import numpy as np


def guardbb(b, m, ntit, btaco, btaco2, btaco3, btaco4, btaco5):
    """Guarda perfiles de Bphi en cinco radios para el análisis temporal."""
    b = np.asarray(b)
    histories = (btaco, btaco2, btaco3, btaco4, btaco5)
    # Los índices corresponden a las capas radiales elegidas para los historiales.
    radial_indices = (21, 22, 29, 39, 49)

    for history, radial_index in zip(histories, radial_indices):
        history[m, :ntit] = b[radial_index, :ntit]

    return histories