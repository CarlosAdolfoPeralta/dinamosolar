
"""Integra el modelo axisimétrico de dínamo con retroalimentación.

Implementación asociada con Sraibman y Minotti (2019), DOI
10.1007/s11207-018-1350-1. Este archivo arma la malla, inicializa los
campos y coordina sus actualizaciones y salidas.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from alfan4 import alfan4
from calca_1 import calca as calca_1
from calcb4_1 import calcb4 as calcb4_1
from densidad import densidad
from etauv4 import etauv4
from grafurut import grafurut
from guardaur import guardaur
from guardautit import guardautit
from guardbb import guardbb
from leap2 import leap2
from omegaf import omegaf
from suavixy import suavixy
from velmeri4 import velmeri4
from ws1alfaBL1 import ws1alfaBL1


plt.set_cmap("jet")
plt.rcParams["savefig.dpi"] = 300

# Parámetros de cierre y resolución del modelo.
# factalf escala el efecto alfa de dínamo (0.35 es el valor de referencia del modelo).
factalf = 0.35
# cs es la constante de Smagorinsky que fija la difusividad turbulenta.
cs = 0.1
# Filtra A y B cada 200 pasos.
suavizado = 200
# Multiplicador de la rotación omega; n=1 corresponde al valor de referencia.
n = 1

# Frecuencias de salida y duración de la integración, medidas en pasos.
dibu = 50000
timesave = 1000
maxpasos = 500000
ntime = maxpasos // timesave
# Paso temporal inicial. El primer ciclo lo sustituye por el límite advectivo/difusivo.
# parameter.m describe 8640 s como la escala histórica, no como el valor inicial actual.
delttime = 1.0

# Campo magnético inicial de referencia en el radio externo, en tesla.
bsol = 1e-4
bcero = bsol

# Dominio estelar: radios en metros; rp limita la penetración del flujo meridional.
rtope = 7e8
rcero = 0.55 * rtope
rp = 0.69 * rtope
# Longitud de promedio asociada al cierre de submalla.
lamda = 0.1 * rcero

# Resolución: 30 intervalos angulares y 63 radiales (31 y 63 puntos, respectivamente).
puntostit = 30
deltit = np.pi / puntostit
puntosr = 63
deltr = (rtope - rcero) / puntosr

# Constantes físicas y amplitud del flujo impuesto.
mu0 = 4 * np.pi * 1e-7
# sigma controla magnitud y signo del flujo.
sigma = -7e-8

dir_name = f"factalf_{factalf}_cs_{cs}_{n}w"
output_dir = Path(__file__).resolve().parent / dir_name
output_dir.mkdir(parents=True, exist_ok=True)
ppath = output_dir
print(f"Directorio de salida: {output_dir}")
pi = np.pi
nx = int(np.ceil((rtope - rcero) / deltr))
ntit = int(np.ceil(pi / deltit)) + 1

# La coordenada angular va del polo norte al polo sur.
r = np.zeros(nx)
tit = np.zeros(ntit)
aprim = np.zeros((nx, ntit))
bprim = np.zeros((nx, ntit))
r[0] = rcero
tit[0] = pi
for i in range(1, nx):
    r[i] = r[i - 1] + deltr
for j in range(1, ntit):
    tit[j] = tit[j - 1] - deltit

a = np.zeros((nx, ntit))
b = np.zeros((nx, ntit))
btit = np.zeros((nx, ntit))
br = np.zeros((nx, ntit))
for name in (
    "brtaco brtaco2 brtaco3 brtaco4 brtaco5 btittaco btittaco2 btittaco3 "
    "btittaco4 btittaco5 btaco btaco2 btaco3 btaco4 btaco5"
).split():
    globals()[name] = np.zeros((ntime, ntit))
for name in "utit3 ur3".split():
    globals()[name] = np.zeros((nx, ntit))
for name in "urt1 urt2 urt3 urt4 urt5 utitt1 utitt2 utitt3 utitt4 utitt5".split():
    globals()[name] = np.zeros((ntime, ntit))
tiempototal = np.zeros(ntime)
ws = np.zeros((nx, ntit))

for i in range(nx):
    for j in range(ntit):
        # Condición inicial del potencial poloidal, proporcional a sin(theta).
        a[i, j] = bcero * 2 * rtope / (2 * rtope - rp) * (r[i] - rp) * np.sin(tit[j])

ur = np.zeros((nx, ntit))
utit = np.zeros((nx, ntit))
omega = omegaf(r, rtope, nx, tit, ntit, n=n)
ro = densidad(r, nx, rtope)
ur2, utit2 = velmeri4(rtope, rcero, rp, nx, ntit, r, tit, sigma, ro)
ur = ur2.copy()
utit = utit2.copy()
alf = alfan4(lamda, r, tit, ur, utit, omega, nx, ntit, deltr, deltit, rtope, factalf)
alfao = alf.copy()
m = 1

while m <= maxpasos:
    graf = 1

    if m % dibu == 0: #Grafico la perturbación de la velocidad poloidal
        grafurut(ur3, utit3, r=r, tit=tit, rtope=rtope)
        plt.savefig(output_dir / f"ur3utit3_t_{m}.png")
        plt.close("all")
        #Grafico las componentes del campo magnético poloidal
        grafurut(br, btit, r=r, tit=tit, rtope=rtope, labels=(r"$B_r$", r"$B_\theta$"))
        plt.savefig(output_dir / f"brbtit_t_{m}.png")
        plt.close("all")
        #Grafico B_r en la tacoclina en función de theta, permite identificar el periodo de inversión del campo magnético
        sample_count = m // timesave - 1
        elapsed_years = np.arange(1, sample_count + 1) * timesave * delttime / (3 * 10**7)
        field_series = brtaco4[:sample_count, 19]
        figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
        axis.plot(elapsed_years, field_series)
        peak_indices, _ = find_peaks(field_series)
        if len(peak_indices) > 1:
            peak_times = elapsed_years[peak_indices]
            peak_values = field_series[peak_indices]
            axis.set_title(
                f"Anios entre picos: {peak_times[-1] - peak_times[-2]:.1f}; "
                f"Cociente entre picos: {peak_values[-1] / peak_values[-2]:.1f}"
            )
        axis.set_xlabel("Años")
        axis.set_ylabel(r"$B_r$")
        figure.savefig(output_dir / f"brtaco_t_{m}.png", dpi=300)
        plt.close("all")
        #Hago el gráfico que muestra la variación de Br en todo valor de theta, en cada tiempo.
        figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
        mesh = axis.contourf(
            elapsed_years,
            np.cos(tit),
            brtaco4[:sample_count, :].T,
            levels=100,
            cmap=plt.get_cmap(),
            antialiased=True,
        )
        axis.set_title("Evolución temporal de Br en la tacoclina")
        axis.set_xlabel("Años")
        axis.set_ylabel(r"$\cos(\theta)$")
        figure.colorbar(mesh, ax=axis, pad=0.04)
        figure.savefig(output_dir / f"brtaco2_t_{m}.png", dpi=600)
        plt.close("all")

    # Estima la difusividad turbulenta local usada en ambas ecuaciones de inducción.
    etau = etauv4(nx, ntit, ur, utit, r, tit, cs, deltr, deltit, omega, br, btit, b, ro)
    if m == 1:
        # El paso temporal inicial respeta límites advectivos y difusivos.
        veltotal = np.zeros((nx, ntit))
        for i in range(nx):
            for j in range(ntit):
                veltotal[i, j] = (
                    np.sqrt(ur[i, j] ** 2 + (omega[i, j] * r[i] * np.sin(tit[j])) ** 2 + utit[i, j] ** 2)
                    + np.sqrt(br[i, j] ** 2 + btit[i, j] ** 2 + b[i, j] ** 2) / np.sqrt(mu0 * ro[i])
                )
        velcondicion = np.max(veltotal)
        etacondicion = np.max(np.abs(etau))
        delttime = min(
            r[0] * deltit / velcondicion,
            (r[0] * deltit) ** 2 / (6 * etacondicion),
            20000,
        )

    # Actualiza el potencial vector poloidal y deriva de él las componentes Br y Btheta.
    a, aprim, br, btit = calca_1(
        a, ntit, nx, deltr, deltit, r, tit, ur, utit, etau, delttime,
        br, btit, b, alf, aprim, m, graf, rtope, omega, lamda, ppath, dibu,
        brtaco=brtaco,
        brtaco2=brtaco2,
        brtaco3=brtaco3,
        brtaco4=brtaco4,
        brtaco5=brtaco5,
        btittaco=btittaco,
        btittaco2=btittaco2,
        btittaco3=btittaco3,
        btittaco4=btittaco4,
        btittaco5=btittaco5,
        timesave=timesave,
    )
    # Actualiza el campo toroidal, incluyendo cizalladura y término subgrid.
    b, bprim = calcb4_1(
        b, ntit, nx, deltr, deltit, r, tit, ur, utit, etau, delttime,
        br, btit, a, omega, bprim, m, 1, lamda, ppath, dibu,
        rtope=rtope,
    )

    # Avance explícito de los campos usando las derivadas temporales calculadas.
    a += delttime * aprim
    b += delttime * bprim
    if m % suavizado == 0:
        b = suavixy(b, nx, ntit)
        a = suavixy(a, nx, ntit)

    if m % timesave == 0:
        print(m)
    m += 1
    # La fuerza magnética induce una circulación que se suma al flujo impuesto.
    ur3, utit3 = leap2(r, tit, nx, ntit, rtope, mu0, omega, b, br, btit, deltr, deltit, ro)
    ur = ur2 + ur3
    utit = utit2 + utit3

    if m > 0:
        # Evoluciona la realimentación vorticidad-alpha del modelo.
        ws, alfBL = ws1alfaBL1(
            lamda, ur2, utit2, ur3, utit3, factalf, ws, delttime, omega,
            r=r, tit=tit, nx=nx, ntit=ntit, deltr=deltr, deltit=deltit,
        )
        alf = alfao + alfBL
    #Reviso si la simulación está corriendo normalmente, o si tuvo inestabilidades numéricas
    if np.isnan(ur[14, 9]):
        raise RuntimeError("Inestabilidad o divergencia")
    #Hago el guardado de los datos
    if m % timesave == 0:
        save_index = m // timesave - 1
        btaco, btaco2, btaco3, btaco4, btaco5 = guardbb(
            b, save_index, ntit, btaco, btaco2, btaco3, btaco4, btaco5
        )
        urt1, urt2, urt3, urt4, urt5 = guardaur(
            ur3, save_index, ntit, urt1, urt2, urt3, urt4, urt5
        )
        utitt1, utitt2, utitt3, utitt4, utitt5 = guardautit(
            utit3, save_index, ntit, utitt1, utitt2, utitt3, utitt4, utitt5
        )
        tiempototal[save_index] = m * delttime
        if m % dibu == 0 or save_index == ntime - 1:
            # Los historiales se guardan en NPZ comprimido, legible con numpy.load.
            np.savez_compressed(
                output_dir / "brtaco.npz",
                tiempototal=tiempototal,
                brtaco=brtaco,
                brtaco2=brtaco2,
                brtaco3=brtaco3,
                brtaco4=brtaco4,
                brtaco5=brtaco5,
            )
            np.savez_compressed(
                output_dir / "btittaco.npz",
                btittaco=btittaco,
                btittaco2=btittaco2,
                btittaco3=btittaco3,
                btittaco4=btittaco4,
                btittaco5=btittaco5,
            )
            np.savez_compressed(
                output_dir / "b.npz",
                btaco=btaco,
                btaco2=btaco2,
                btaco3=btaco3,
                btaco4=btaco4,
                btaco5=btaco5,
            )
            np.savez_compressed(
                output_dir / "ur.npz",
                urt1=urt1,
                urt2=urt2,
                urt3=urt3,
                urt4=urt4,
                urt5=urt5,
            )
            np.savez_compressed(
                output_dir / "utit.npz",
                utitt1=utitt1,
                utitt2=utitt2,
                utitt3=utitt3,
                utitt4=utitt4,
                utitt5=utitt5,
            )
            state = {
                name: value
                for name, value in globals().items()
                if isinstance(value, (np.ndarray, int, float, str, np.number))
            }
            np.savez_compressed(output_dir / "todo.npz", **state)