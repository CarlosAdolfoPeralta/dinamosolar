import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def grafurut(ur, utit, *, r, tit, rtope=None, labels=(r"$u_r$", r"$u_\theta$")):
    """Dibuja dos componentes sobre la sección meridional normalizada."""
    import matplotlib.pyplot as plt

    ur = np.asarray(ur)
    utit = np.asarray(utit)
    r = np.asarray(r)
    tit = np.asarray(tit)
    if rtope is None:
        rtope = np.max(r)
    # Proyección esférica al plano (x,z), con distancias en unidades de rtope.
    x = r[:, None] * np.sin(tit)[None, :] / rtope
    y = r[:, None] * np.cos(tit)[None, :] / rtope

    figure, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for axis, field, label in zip(axes, (ur, utit), labels):
        image = axis.pcolormesh(x, y, field, shading="gouraud")
        axis.set_title(label)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(r"$x/r_{\rm tope}$")
        axis.set_ylabel(r"$y/r_{\rm tope}$")
        divider = make_axes_locatable(axis)
        colorbar_axis = divider.append_axes("right", size="4%", pad=0.08)
        figure.colorbar(image, cax=colorbar_axis)

    if labels[0] == r"$B_r$":
        figure.suptitle("Componentes del campo magnético poloidal")
    else:
        figure.suptitle("Componentes de la perturbación a la velocidad meridional")

    return figure, axes