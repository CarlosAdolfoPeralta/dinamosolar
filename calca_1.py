from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _jit import njit
from mpl_toolkits.axes_grid1 import make_axes_locatable

from calcanx import calcanx
from laplaciano import laplaciano


def _plot_calca(fields, titles, output_path, r, tit, rtope):
    x = r[:, None] * np.sin(tit)[None, :] / rtope
    y = r[:, None] * np.cos(tit)[None, :] / rtope
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for index, (axis, field, title) in enumerate(zip(axes.flat, fields, titles)):
        if index == 0:
            image = axis.contour(x, y, field, levels=20, cmap="jet")
        else:
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
def _calca_compute(
    a, aprim, nx, ntit, deltr, deltit, delttime, r, tit, ur, utit, eta, br,
    btit, b, alfa, rtope, omega, lamda,
):
    """Calcula la derivada temporal del potencial poloidal A."""
    rt = 0.72 * rtope
    aprimer = np.zeros((nx, ntit))
    asegundo = np.zeros((nx, ntit))
    atercer = np.zeros((nx, ntit))
    acuarto = np.zeros((nx, ntit))

    a[:, 0] = 0
    a[:, ntit - 1] = 0
    a[0, :] = 0

    # En la zona convectiva, A determina Br y Btheta mediante rot(A e_phi).
    for i in range(1, nx - 1):
        for j in range(1, ntit - 1):
            if r[i] > rt:
                dadr = (a[i + 1, j] - a[i - 1, j]) / (2 * deltr)
                dadtit = (a[i, j + 1] - a[i, j - 1]) / (2 * deltit)
                btit[i, j] = -a[i, j] / r[i] - dadr
                br[i, j] = dadtit / r[i] + np.cos(tit[j]) / np.sin(tit[j]) / r[i] * a[i, j]

    # RHS: transporte, difusión turbulenta, cierre subgrid y efecto alpha B.
    for i in range(1, nx - 1):
        for j in range(1, ntit - 1):
            if r[i] > rt:
                au1 = ur[i, j] * btit[i, j] - utit[i, j] * br[i, j]
                aprimer[i, j] = au1

                dadr = (a[i + 1, j] - a[i - 1, j]) / (2 * deltr)
                dadtit = (a[i, j + 1] - a[i, j - 1]) / (r[i] * 2 * deltit)
                d2adr2 = (a[i + 1, j] - 2 * a[i, j] + a[i - 1, j]) / deltr**2
                d2adtit2 = (a[i, j + 1] - 2 * a[i, j] + a[i, j - 1]) / deltit**2
                inverse_r = 1 / r[i]
                inverse_sine = 1 / np.sin(tit[j])
                au2 = (
                    d2adr2
                    + 2 * inverse_r * dadr
                    + inverse_r**2 * d2adtit2
                    + inverse_r**2 * inverse_sine * np.cos(tit[j]) * dadtit
                    - (inverse_r * inverse_sine) ** 2 * a[i, j]
                )
                asegundo[i, j] = au2 * eta[i, j]

            durdr = (ur[i + 1, j] - ur[i - 1, j]) / (2 * deltr)
            durdtit = (ur[i, j + 1] - ur[i, j - 1]) / (r[i] * 2 * deltit)
            dutitdr = (utit[i + 1, j] - utit[i - 1, j]) / (2 * deltr)
            dutitdtit = (utit[i, j + 1] - utit[i, j - 1]) / (r[i] * 2 * deltit)
            dbrdr = (br[i + 1, j] - br[i - 1, j]) / (2 * deltr)
            dbrdtit = (br[i, j + 1] - br[i, j - 1]) / (r[i] * 2 * deltit)
            dbtitdr = (btit[i + 1, j] - btit[i - 1, j]) / (2 * deltr)
            dbtitdtit = (btit[i, j + 1] - btit[i, j - 1]) / (r[i] * 2 * deltit)

            au31 = durdr * dbtitdr + durdtit * dbtitdtit
            au32 = dutitdr * dbrdr + dutitdtit * dbrdtit
            au33 = (ur[i, j] + dutitdtit * r[i]) * btit[i, j] / r[i] ** 2
            au34 = durdtit * br[i, j] / r[i]
            au35 = dbrdtit * ur[i, j] / r[i]
            au36 = (br[i, j] + r[i] * dbtitdtit) * utit[i, j] / r[i] ** 2
            atercer[i, j] = (au31 - au32 + au33 + au34 - au35 - au36) * lamda**2 / 24
            acuarto[i, j] = alfa[i, j] * b[i, j]

    # Suaviza el aporte subgrid para regularizar la solución en la malla.
    for _ in range(10):
        atercer = laplaciano(atercer, lamda, nx, ntit, deltr, deltit, r, tit)

    for i in range(1, nx - 1):
        for j in range(1, ntit - 1):
            aprim[i, j] = aprimer[i, j] + asegundo[i, j] + atercer[i, j] + acuarto[i, j]

    # En el borde externo se sustituye la diferencia centrada por una unilateral.
    aprimer, atercer, acuarto = calcanx(
        a, ntit, nx, deltr, deltit, r, tit, ur, utit, eta, delttime, br, btit, b,
        alfa, aprimer, atercer, acuarto, lamda,
    )
    for j in range(1, ntit - 1):
        aprim[nx - 1, j] = aprimer[nx - 1, j] + atercer[nx - 1, j] + acuarto[nx - 1, j]

    return a, aprim, br, btit, aprimer, asegundo, atercer, acuarto


def calca(
    a,
    ntit,
    nx,
    deltr,
    deltit,
    r,
    tit,
    ur,
    utit,
    eta,
    delttime,
    br,
    btit,
    b,
    alfa,
    aprim,
    m,
    graf,
    rtope,
    omega,
    lamda,
    ppath,
    dibu,
    *,
    brtaco,
    brtaco2,
    brtaco3,
    brtaco4,
    brtaco5,
    btittaco,
    btittaco2,
    btittaco3,
    btittaco4,
    btittaco5,
    timesave,
):
    a = np.asarray(a)
    aprim = np.asarray(aprim)
    ur = np.asarray(ur)
    utit = np.asarray(utit)
    eta = np.asarray(eta)
    br = np.asarray(br)
    btit = np.asarray(btit)
    b = np.asarray(b)
    alfa = np.asarray(alfa)
    omega = np.asarray(omega)
    r = np.asarray(r)
    tit = np.asarray(tit)

    a, aprim, br, btit, aprimer, asegundo, atercer, acuarto = _calca_compute(
        a, aprim, nx, ntit, deltr, deltit, delttime, r, tit, ur, utit, eta,
        br, btit, b, alfa, rtope, omega, lamda,
    )

    if m > 1 and m % timesave == 0:
        save_index = m // timesave - 1
        brtaco[save_index, :] = br[24, :]
        brtaco2[save_index, :] = br[25, :]
        brtaco3[save_index, :] = br[29, :]
        brtaco4[save_index, :] = br[39, :]
        brtaco5[save_index, :] = br[49, :]
        btittaco[save_index, :] = btit[21, :]
        btittaco2[save_index, :] = btit[22, :]
        btittaco3[save_index, :] = btit[29, :]
        btittaco4[save_index, :] = btit[39, :]
        btittaco5[save_index, :] = btit[49, :]
    elif m <= 1:
        for history in (
            brtaco, brtaco2, brtaco3, brtaco4, brtaco5,
            btittaco, btittaco2, btittaco3, btittaco4, btittaco5,
        ):
            history[0, :] = 0

    psi = a * r[:, None] * np.sin(tit)[None, :]
    if m % dibu == 0 and graf == 1:
        plot_dir = Path(ppath)
        if not plot_dir.is_absolute():
            plot_dir = Path.cwd() / str(ppath).strip("/\\")
        path = plot_dir / f"a_t_{m}.png"
        _plot_calca(
            (psi, aprim, aprimer, asegundo, acuarto, eta),
            ("Líneas de campo magnético", "Derivada temporal del potencial vector", "Término convectivo", "Término disipativo", "Término de dínamo", "Difusividad magnética"),
            path,
            r,
            tit,
            rtope,
        )

    return a, aprim, br, btit