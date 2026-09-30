# dinamosolar
MODELO AXISIMETRICO DE DINAMO SOLAR
====================================

PROPOSITO
---------
Este programa integra un modelo axisimetrico de dínamo a gran escala en 
coordenadas esféricas. Evoluciona el campo magnético poloidal y toroidal, incluye 
la producción de campo por rotación diferencial y un efecto alfa, y permite que
la fuerza magnética modifique el flujo meridional. La finalidad es estudiar la
evolución temporal y la retroalimentación entre el campo y los movimientos del
fluido estelar.

El alcance científico corresponde al modelo descrito por Sraibman y Minotti
(2019). La secuencia de cálculo y los nombres de los términos que siguen se
refieren a la implementación de esta carpeta; no sustituyen la derivación
completa del artículo.

ORGANIZACION DEL CALCULO
------------------------
1. toy_prueba.py crea la malla radial y angular, inicializa el potencial
   poloidal, el perfil de rotación y el flujo meridional de fondo, y coordina
   la integración temporal. Este es el archivo principal. Se puede ejecutar
   directamente como está, y permite estudiar el caso con parámetros solares.
2. alfan4.py calcula el coeficiente alfa cinemático a partir de los gradientes
   de la rotación diferencial. ws1alfaBL1.py evoluciona una contribución alfa
   adicional asociada a la retroalimentación.
3. etauv4.py estima una difusividad turbulenta local usando el coeficiente de
   Smagorinsky, la deformación del flujo, la rotación y el campo magnético.
4. calca_1.py integra la ecuación del potencial vector poloidal A. De A obtiene
   las componentes Br y Btheta; su ecuación reúne transporte, difusión,
   contribuciones subgrid y la fuente alfa B.
5. calcb4_1.py integra la ecuación del campo toroidal Bphi. Incluye transporte,
   difusión, cizalladura por rotación diferencial y el cierre subgrid Scero3.py.
6. leap2.py calcula la perturbación del flujo meridional inducida por la fuerza
   magnética y la suma al flujo de fondo definido por velmeri4.py.
7. toy_prueba.py avanza los campos en el tiempo, aplica suavizado cada cierto
   número de pasos y verifica si aparece una inestabilidad numérica.

Los kernels numericos usan Numba si está instalado. _jit.py conserva una
alternativa interpretada si Numba no está disponible.

PARAMETROS PRINCIPALES
----------------------
factalf: amplitud relativa del efecto alfa; valor usado: 0.35.
cs: coeficiente de Smagorinsky para la difusividad turbulenta; valor: 0.1.
suavizado: intervalo de suavizado de A y B; valor: 200 pasos.
n: multiplicador de la rotación omega; valor: 1.
dibu: intervalo entre gráficos de diagnóstico; valor: 50000 pasos.
timesave: intervalo de muestreo de historiales; valor: 1000 pasos.
maxpasos: duración máxima de la corrida; valor: 500000 pasos.
delttime: paso inicial; se recalcula al comienzo con límites advectivos y
  difusivos.
rtope: radio externo de la malla, 7e8 m.
rcero: radio interno del dominio, 0.55 rtope.
rp: radio hasta el que se permite penetrar al flujo meridional, 0.69 rtope.
lamda: longitud asociada al promedio subgrid, 0.1 rcero.
puntostit: cantidad de intervalos angulares; deltit = pi/puntostit.
puntosr: cantidad de intervalos radiales; puntosr = 63.
mu0: permeabilidad magnética del vacío.
sigma: magnitud y signo del flujo meridional impuesto; en toy_prueba.py vale
  -7e-8.

ARCHIVOS AUXILIARES
-------------------
calca_1.py y calcanx.py: ecuación poloidal y tratamiento del borde externo.
calcb4_1.py: ecuación toroidal y gráficos de sus términos.
Scero3.py: término subgrid de la ecuación toroidal.
etauv4.py: estimación de la difusividad turbulenta.
omegaf.py y densidad.py: perfiles de rotación diferencial y densidad.
velmeri4.py: flujo meridional de fondo.
leap2.py: flujo meridional inducido por el campo magnético.
ws1alfaBL1.py: evolución de la vorticidad de retroalimentación y del alfa BL.
suavixy.py, suavixy2d.py y suavixy2dbl.py: filtros y simetría ecuatorial.
guardbb.py, guardaur.py y guardautit.py: muestreo temporal en radios seleccionados.
grafurut.py: gráficos de componentes en el plano meridional.

SALIDAS
-------
La carpeta de salida se crea junto a toy_prueba.py con el nombre construido a
partir de factalf, cs y n. Los historiales se guardan como archivos NumPy NPZ
comprimidos:

brtaco.npz: tiempototal y cinco historiales de Br.
btittaco.npz: cinco historiales de Btheta.
b.npz: cinco historiales del campo toroidal.
ur.npz: cinco historiales de la perturbación ur.
utit.npz: cinco historiales de la perturbación utheta.
todo.npz: estado y parámetros serializables de la corrida.

Las series guardaur, guardautit y guardbb corresponden a cinco índices
radiales (21, 22, 29, 39 y 49), no a una malla radial completa. Los datos de
cada fila se registran cada timesave pasos. Los archivos se abren, por ejemplo,
con:

    with numpy.load("ur.npz") as datos:
        ur_en_radio_3 = datos["urt3"]


REQUISITOS
----------
Python, NumPy, SciPy y Matplotlib. Numba es opcional: acelera los kernels
numericos cuando está instalado.

REFERENCIA
----------
Sraibman, L., & Minotti, F. (2019). Large-scale model of the axisymmetric
dynamo with feedback effects. Solar Physics, 294(1), 14.
DOI: https://doi.org/10.1007/s11207-018-1350-1

