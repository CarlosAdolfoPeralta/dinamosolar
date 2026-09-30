# MODELO AXISIMÉTRICO DE DINAMO SOLAR

======================================

## PROPÓSITO

---

Este programa implementa un modelo axisimétrico de dinamo a gran escala en coordenadas esféricas. El modelo evoluciona los campos magnéticos poloidal y toroidal, incluye la generación de campo debida a la rotación diferencial y al efecto alfa, y permite que la fuerza magnética modifique el flujo meridional. La finalidad es estudiar la evolución temporal del campo magnético y la retroalimentación entre el campo y los movimientos del fluido estelar.

El alcance científico del modelo corresponde al trabajo de Sraibman y Minotti (2019). La secuencia de cálculo y los nombres de los términos utilizados a continuación se refieren específicamente a la implementación de esta carpeta y no sustituyen la derivación completa presentada en el artículo.

## ORGANIZACIÓN DEL CÁLCULO

---

1. `toy_prueba.py` crea las mallas radial y angular, inicializa el potencial poloidal, el perfil de rotación y el flujo meridional de fondo, y coordina la integración temporal. Este es el archivo principal del programa. Puede ejecutarse directamente con los parámetros definidos en el código y permite estudiar el caso correspondiente a los parámetros solares.

2. `alfan4.py` calcula el coeficiente alfa cinemático a partir de los gradientes de la rotación diferencial. `ws1alfaBL1.py` evoluciona una contribución alfa adicional asociada a la retroalimentación.

3. `etauv4.py` estima la difusividad turbulenta local utilizando el coeficiente de Smagorinsky, la deformación del flujo, la rotación y el campo magnético.

4. `calca_1.py` integra la ecuación del potencial vectorial poloidal `A`. A partir de `A` se obtienen las componentes `Br` y `Btheta`. La ecuación incluye términos de transporte, difusión, contribuciones subgrid y la fuente alfa `B`.

5. `calcb4_1.py` integra la ecuación del campo toroidal `Bphi`. Incluye términos de transporte, difusión y cizalladura debida a la rotación diferencial, además del cierre subgrid definido en `Scero3.py`.

6. `leap2.py` calcula la perturbación del flujo meridional inducida por la fuerza magnética y la suma al flujo meridional de fondo definido por `velmeri4.py`.

7. `toy_prueba.py` avanza los campos en el tiempo, aplica el suavizado cada cierto número de pasos y verifica la aparición de posibles inestabilidades numéricas.

Los kernels numéricos utilizan Numba cuando está instalado. `_jit.py` proporciona una alternativa interpretada para ejecutar estos kernels cuando Numba no está disponible.

## PARÁMETROS PRINCIPALES

---

Los principales parámetros utilizados por la simulación son los siguientes:

* `factalf`: amplitud relativa del efecto alfa. Valor utilizado: 0.35.

* `cs`: coeficiente de Smagorinsky utilizado para calcular la difusividad turbulenta. Valor utilizado: 0.1.

* `suavizado`: intervalo entre aplicaciones del suavizado de `A` y `B`. Valor utilizado: 200 pasos.

* `n`: multiplicador de la rotación `omega`. Valor utilizado: 1.

* `dibu`: intervalo entre gráficos de diagnóstico. Valor utilizado: 50000 pasos.

* `timesave`: intervalo de muestreo de los historiales. Valor utilizado: 1000 pasos.

* `maxpasos`: número máximo de pasos de la simulación. Valor utilizado: 500000 pasos.

* `delttime`: paso temporal inicial. Se recalcula al comienzo de la simulación de acuerdo con los límites advectivos y difusivos.

* `rtope`: radio externo de la malla. Valor utilizado: `7e8 m`.

* `rcero`: radio interno del dominio. Valor utilizado: `0.55*rtope`.

* `rp`: radio hasta el cual se permite la penetración del flujo meridional. Valor utilizado: `0.69*rtope`.

* `lamda`: longitud característica asociada al promedio subgrid. Valor utilizado: `0.1*rcero`.

* `puntostit`: número de intervalos angulares. El tamaño del intervalo angular se define como `deltit = pi/puntostit`.

* `puntosr`: número de intervalos radiales. Valor utilizado: 63.

* `mu0`: permeabilidad magnética del vacío.

* `sigma`: magnitud y signo del flujo meridional impuesto. En `toy_prueba.py`, su valor es `-7e-8`.

## ARCHIVOS AUXILIARES

---

* `calca_1.py` y `calcanx.py`: calculan la evolución del campo poloidal y tratan los términos asociados al borde externo.

* `calcb4_1.py`: calcula la evolución del campo toroidal y genera gráficos de los distintos términos de su ecuación.

* `Scero3.py`: calcula el término subgrid de la ecuación del campo toroidal.

* `etauv4.py`: estima la difusividad turbulenta local.

* `omegaf.py` y `densidad.py`: calculan los perfiles de rotación diferencial y densidad, respectivamente.

* `velmeri4.py`: calcula el flujo meridional de fondo.

* `leap2.py`: calcula la perturbación del flujo meridional inducida por el campo magnético.

* `ws1alfaBL1.py`: evoluciona la vorticidad de retroalimentación y el término alfa asociado al mecanismo BL.

* `suavixy.py`, `suavixy2d.py` y `suavixy2dbl.py`: aplican filtros y las condiciones de simetría ecuatorial utilizadas en el modelo.

* `guardbb.py`, `guardaur.py` y `guardautit.py`: realizan el muestreo temporal en radios seleccionados.

* `grafurut.py`: genera gráficos de las componentes del campo y del flujo en el plano meridional.

## SALIDAS

---

La carpeta de salida se crea en el mismo directorio que `toy_prueba.py`. Su nombre se construye a partir de los valores de `factalf`, `cs` y `n`.

Los historiales se guardan como archivos NumPy `.npz` comprimidos:

* `brtaco.npz`: contiene `tiempototal` y cinco historiales de `Br`.

* `btittaco.npz`: contiene cinco historiales de `Btheta`.

* `b.npz`: contiene cinco historiales del campo toroidal.

* `ur.npz`: contiene cinco historiales de la perturbación `ur`.

* `utit.npz`: contiene cinco historiales de la perturbación `utheta`.

* `todo.npz`: contiene el estado y los parámetros serializables de la simulación.

Las series generadas por `guardaur`, `guardautit` y `guardbb` corresponden a cinco índices radiales específicos: 21, 22, 29, 39 y 49. Por lo tanto, estos historiales no representan una malla radial completa.

Los datos de cada historial se registran cada `timesave` pasos.

Los archivos `.npz` pueden abrirse, por ejemplo, mediante:

```python
with numpy.load("ur.npz") as datos:
    ur_en_radio_3 = datos["urt3"]
```

## REQUISITOS

---

Se requiere Python junto con las bibliotecas **NumPy**, **SciPy** y **Matplotlib**.

**Numba** es opcional. Cuando está instalado, se utiliza para acelerar los kernels numéricos. Si Numba no está disponible, `_jit.py` proporciona una alternativa interpretada.

## REFERENCIA

---

Sraibman, L., & Minotti, F. (2019). *Large-scale model of the axisymmetric dynamo with feedback effects*. Solar Physics, 294(1), 14.

DOI: https://doi.org/10.1007/s11207-018-1350-1


