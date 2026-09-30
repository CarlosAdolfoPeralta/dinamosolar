from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _jit import njit
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Scero3 import Scero3


def _plot_calcb(fields, titles, output_path, r, tit, rtope):
    x = r[:, None] * np.sin(tit)[None, :] / rtope
    y = r[:, None] * np.cos(tit)[None, :] / rtope
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for axis, field, title in zip(axes.flat, fields, titles):
        image = axis.pcolormesh(x, y, field, shading="gouraud")
        axis.set_title(title, pad=16)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(r"$x/r_{\rm tope}$")
        axis.set_ylabel(r"$y/r_{\rm tope}$")
        divider = make_axes_locatable(axis)
        colorbar_axis = divider.append_axes("right", size="4%", pad=0.08)
        figure.colorbar(image, cax=colorbar_axis)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


@njit(cache=True)
def _calcb_compute(
    b, bprim, ntit, nx, deltr, deltit, r, tit, ur, utit, etat, br, btit,
    a, omega, m, lamda,
):
    """Calcula la derivada temporal del campo magnético toroidal Bphi."""
    bprimer = np.zeros((nx, ntit))
    bsegundo = np.zeros((nx, ntit))
    btercer = np.zeros((nx, ntit))
    bcuarto = np.zeros((nx, ntit))
    scero = np.zeros((nx, ntit))

    b[:, 0] = 0
    b[:, ntit - 1] = 0
    b[0, :] = 0
    b[nx - 1, :] = 0

    # La ecuación combina transporte, difusión, Ω-efecto y gradientes de eta.
    for i in range(1, nx - 1):
        for j in range(1, ntit - 1):
            durdr = (ur[i + 1, j] - ur[i - 1, j]) / (2 * deltr)
            dutitdtit = (utit[i, j + 1] - utit[i, j - 1]) / (2 * deltit)
            dbdr = (b[i + 1, j] - b[i - 1, j]) / (2 * deltr)
            dbdtit = (b[i, j + 1] - b[i, j - 1]) / (2 * deltit)
            bprimer[i, j] = -(1 / r[i]) * (
                ur[i, j] * b[i, j] + r[i] * durdr * b[i, j] + r[i] * ur[i, j] * dbdr
                + dutitdtit * b[i, j] + utit[i, j] * dbdtit
            )
            d2bdr2 = (b[i + 1, j] - 2 * b[i, j] + b[i - 1, j]) / deltr**2
            d2bdtit2 = (b[i, j + 1] - 2 * b[i, j] + b[i, j - 1]) / deltit**2
            sine = np.sin(tit[j])
            cotangent = np.cos(tit[j]) / sine
            bsegundo[i, j] = (
                2 * dbdr / r[i] + d2bdr2 + cotangent * dbdtit / r[i] ** 2
                + d2bdtit2 / r[i] ** 2 - b[i, j] / (r[i] ** 2 * sine**2)
            ) * etat[i, j]
            dadtit = (a[i, j + 1] - a[i, j - 1]) / (2 * deltit)
            dadr = (a[i + 1, j] - a[i - 1, j]) / (2 * deltr)
            domegadr = (omega[i + 1, j] - omega[i - 1, j]) / (2 * deltr)
            domegadtit = (omega[i, j + 1] - omega[i, j - 1]) / (2 * deltit)
            btercer[i, j] = (
                (np.cos(tit[j]) * a[i, j] + sine * dadtit) * domegadr
                + (-r[i] * dadr - a[i, j]) * sine / r[i] * domegadtit
            )
            detadr = (etat[i + 1, j] - etat[i - 1, j]) / (2 * deltr)
            detadtit = (etat[i, j + 1] - etat[i, j - 1]) / (2 * deltit)
            bcuarto[i, j] = (
                detadr / r[i] * b[i, j] + detadr * dbdr
                + detadtit / r[i] ** 2 * (dbdtit + b[i, j] * cotangent)
            )

    if m > 1:
        scero = Scero3(
            b, lamda, ur, utit, omega, br, btit, r, tit, nx, ntit, deltr, deltit,
        )

    for i in range(1, nx - 1):
        for j in range(1, ntit - 1):
            bprim[i, j] = bprimer[i, j] + bsegundo[i, j] + btercer[i, j] + bcuarto[i, j] + scero[i, j]

    return b, bprim, bsegundo, btercer, bcuarto, scero


def calcb4(
    b,
    ntit,
    nx,
    deltr,
    deltit,
    r,
    tit,
    ur,
    utit,
    etat,
    delttime,
    br,
    btit,
    a,
    omega,
    bprim,
    m,
    graf,
    lamda,
    ppath,
    dibu,
    *,
    rtope=None,
):
    b = np.asarray(b)
    bprim = np.asarray(bprim)
    r = np.asarray(r)
    tit = np.asarray(tit)
    ur = np.asarray(ur)
    utit = np.asarray(utit)
    etat = np.asarray(etat)
    br = np.asarray(br)
    btit = np.asarray(btit)
    a = np.asarray(a)
    omega = np.asarray(omega)

    b, bprim, bsegundo, btercer, bcuarto, scero = _calcb_compute(
        b, bprim, ntit, nx, deltr, deltit, r, tit, ur, utit, etat, br, btit,
        a, omega, m, lamda,
    )

    if m % dibu == 0 and graf == 1:
        plot_dir = Path(ppath)
        if not plot_dir.is_absolute():
            plot_dir = Path.cwd() / str(ppath).strip("/\\")
        path = plot_dir / f"b2_t_{m}.png"
        _plot_calcb(
            (b, bprim, bsegundo, btercer, bcuarto, scero),
            (
                "Campo magnético toroidal",
                "Derivada temporal del campo magnético toroidal",
                "Término disipativo",
                r"$(\nabla \times \mathbf{A}) \cdot \boldsymbol{\Omega}$",
                r"$\nabla \times (\eta \, \nabla \times \mathbf{B})$",
                "Término de subgrilla",
            ),
            path,
            r,
            tit,
            rtope if rtope is not None else r[-1] + deltr,
        )

    return b, bprim