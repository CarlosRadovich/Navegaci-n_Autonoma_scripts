<!--
  Actividad 2.1 — Seguimiento Autonomo de Carril
  Navegacion Autonoma (MR4010.10)
  Instituto Tecnologico y de Estudios Superiores de Monterrey
-->

<div align="center">

# Seguimiento Autonomo de Carril mediante Vision Computacional y Control PID

**Actividad 2.1 — Navegacion Autonoma (MR4010.10)**

Instituto Tecnologico y de Estudios Superiores de Monterrey
Maestria en Inteligencia Artificial

---

| Alumno | Matricula |
|:---|:---:|
| Antonio Olvera Donlucas | A01795617 |
| Carlos Monir Radovich Saad | A01797569 |
| Andres Roberto Osuna Gonzalez | A01796264 |
| Oscar Alberto Ramirez Anaya | A01795438 |

**Fecha de entrega:** Mayo 2026

---

</div>

## Tabla de Contenidos

- [Resumen](#resumen)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Pipeline de Vision Computacional](#pipeline-de-vision-computacional)
- [Controlador PID](#controlador-pid)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Descripcion de los Scripts](#descripcion-de-los-scripts)
- [Requisitos del Entorno](#requisitos-del-entorno)
- [Instrucciones de Ejecucion](#instrucciones-de-ejecucion)
- [Parametros y Configuracion](#parametros-y-configuracion)
- [Resultados y Observaciones](#resultados-y-observaciones)
- [Decisiones de Diseno](#decisiones-de-diseno)
- [Trabajo Futuro](#trabajo-futuro)
- [Referencias](#referencias)
- [Licencia](#licencia)

---

## Resumen

Este proyecto implementa un sistema de **seguimiento autonomo de carril** para un vehiculo simulado en [Webots](https://cyberbotics.com/). El sistema utiliza una camara a bordo del vehiculo (BMW X5), procesa la imagen en tiempo real mediante tecnicas de vision computacional (OpenCV), y ajusta automaticamente la direccion del vehiculo usando un controlador PID.

El desarrollo sigue una metodologia incremental: cada script representa una etapa del pipeline, desde la captura basica de imagen hasta el controlador PID completo con suavizado exponencial del error, validado a 50 km/h.

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                     SIMULADOR WEBOTS                            │
│                                                                 │
│   ┌──────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│   │  Camara  │───>│  Pipeline Vision  │───>│  Controlador PID │  │
│   │  a bordo │    │   (OpenCV)        │    │                  │  │
│   └──────────┘    └──────────────────┘    └────────┬─────────┘  │
│                                                     │            │
│   ┌──────────┐                            ┌────────▼─────────┐  │
│   │ Display  │<───────────────────────────│  Actuadores      │  │
│   │ onboard  │                            │  (direccion,vel) │  │
│   └──────────┘                            └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pipeline de Vision Computacional

El pipeline de procesamiento de imagen consta de las siguientes etapas:

```
Imagen cruda (128x64, 4 canales)
        │
        ▼
  1. Escala de grises (cvtColor BGR2GRAY)
        │
        ▼
  2. Suavizado Gaussiano (kernel 5x5)
        │
        ▼
  3. Deteccion de bordes (Canny, umbrales 50/150)
        │
        ▼
  4. Region de interes (mascara trapezoidal, fillPoly)
        │
        ▼
  5. Deteccion de lineas (HoughLinesP)
        │
        ▼
  6. Calculo de error lateral (desviacion del centro)
        │
        ▼
  7. Controlador PID → angulo de direccion
```

### Justificacion de umbrales

Los umbrales de Canny se establecieron en **50/150** en lugar de los valores tipicos (80/200). Esto se debe a que la imagen de la camara de Webots es de resolucion baja (128x64 pixeles), y con umbrales mas altos se perdia la deteccion de los bordes del carril.

---

## Controlador PID

El controlador PID calcula el angulo de direccion a partir del error lateral entre el centro de la imagen y la linea del carril mas cercana.

### Ecuacion del controlador

```
u(t) = Kp * e(t) + Ki * integral(e) + Kd * de/dt
```

### Ganancias por version

| Version | Kp | Ki | Kd | Velocidad | Notas |
|:---|:---:|:---:|:---:|:---:|:---|
| Paso 4 (10 km/h) | 0.001 | 0.0 | 0.0 | 10 km/h | Proporcional puro, primera prueba estable |
| Paso 5 (50 km/h) | 0.0007 | 0.0 | 0.0 | 50 km/h | Kp reducido para menor agresividad |
| Paso 6 (final) | 0.008 | 0.0 | 0.015 | 50 km/h | Con derivativo y suavizado EMA |

### Suavizado exponencial del error (EMA)

La version final (Paso 6) implementa un promedio movil exponencial para reducir la sensibilidad al ruido:

```
error_suavizado = alpha * error_anterior + (1 - alpha) * error_nuevo
```

Con `alpha = 0.6`, se da 60% de peso al valor anterior y 40% al nuevo, lo que filtra cambios bruscos entre frames consecutivos.

---

## Estructura del Repositorio

```
actividad_2_1_lane_following/
├── README.md                           # Este documento
├── LICENSE                             # Licencia Apache 2.0
├── .gitignore                          # Archivos excluidos del repositorio
└── src/
    ├── 01_grayscale_controller.py      # Paso 1: Conversion a grises
    ├── 02_canny_edge_detection.py      # Paso 2: Deteccion de bordes Canny
    ├── 03_hough_line_detection.py      # Paso 3: Deteccion de lineas Hough
    ├── 04_pid_lane_following.py        # Paso 4: PID a 10 km/h
    ├── 05_pid_50kmh_validation.py      # Paso 5: Validacion PID a 50 km/h
    └── 06_pid_smoothed_final.py        # Paso 6: PID con suavizado (final)
```

---

## Descripcion de los Scripts

### `01_grayscale_controller.py` — Controlador base con grises

Captura la imagen de la camara, la convierte a escala de grises y la muestra en el display de Webots. El vehiculo se controla manualmente con el teclado (flechas para direccion/velocidad, tecla `A` para captura de pantalla).

### `02_canny_edge_detection.py` — Deteccion de bordes

Agrega al pipeline la deteccion de bordes con el algoritmo de Canny, precedida por suavizado Gaussiano para reducir ruido. El resultado se muestra en el display para validacion visual.

### `03_hough_line_detection.py` — Deteccion de lineas

Incorpora la region de interes trapezoidal (ROI) y la transformada de Hough probabilistica (HoughLinesP) para detectar segmentos de linea recta. Las lineas detectadas se dibujan sobre imagen negra y se muestran en el display.

### `04_pid_lane_following.py` — Controlador PID (10 km/h)

Integra el pipeline completo con un controlador PID que ajusta automaticamente la direccion. Usa ganancias conservadoras (solo proporcional) y suavizado del cambio de angulo para evitar oscilaciones. Primera prueba de conduccion autonoma a velocidad baja.

### `05_pid_50kmh_validation.py` — Validacion a 50 km/h

Misma arquitectura que el Paso 4 con ganancias PID mas conservadoras (`Kp = 0.0007`) y menor cambio maximo de direccion por paso (`0.025 rad`). Valida el requisito minimo de velocidad de la actividad.

### `06_pid_smoothed_final.py` — Version final con suavizado EMA

Version final que incorpora suavizado exponencial del error, ganancia derivativa activa (`Kd = 0.015`), manejo de racha sin deteccion de lineas, y anti-windup en el termino integral. Parametros de Hough ajustados para imagenes pequenas (`threshold=15`, `minLen=8`, `maxGap=5`).

---

## Requisitos del Entorno

### Software

| Componente | Version | Proposito |
|:---|:---|:---|
| [Webots](https://cyberbotics.com/) | R2023b o superior | Simulador de robotica |
| Python | 3.8+ | Lenguaje del controlador |
| NumPy | 1.21+ | Manipulacion de arreglos de imagen |
| OpenCV (cv2) | 4.5+ | Procesamiento de imagen |

### Modelo de simulacion

- **Vehiculo:** BMW X5 (incluido en Webots)
- **Camara:** Dispositivo `camera` (128x64 px, 4 canales)
- **Display:** Dispositivo `display_image` (visualizacion del pipeline)
- **Escenario:** Pista con marcas de carril visibles

---

## Instrucciones de Ejecucion

### 1. Abrir el mundo de Webots

Abrir el archivo `.wbt` del escenario de la actividad en Webots.

### 2. Configurar el controlador

En las propiedades del nodo `Robot` (BMW X5), establecer el campo `controller` con el nombre del script deseado. Por ejemplo:

```
controller: "06_pid_smoothed_final"
```

### 3. Copiar el script al directorio de controladores

Copiar el archivo `.py` correspondiente al directorio de controladores del proyecto de Webots:

```bash
cp src/06_pid_smoothed_final.py <ruta_proyecto_webots>/controllers/06_pid_smoothed_final/
```

### 4. Ejecutar la simulacion

Presionar el boton de **Play** en Webots. La consola mostrara informacion de depuracion con el estado del controlador.

### Controles del teclado (scripts con control manual)

| Tecla | Accion |
|:---:|:---|
| Flecha arriba | Incrementar velocidad (+5 km/h) |
| Flecha abajo | Decrementar velocidad (-5 km/h) |
| Flecha derecha | Girar a la derecha (+0.05 rad) |
| Flecha izquierda | Girar a la izquierda (-0.05 rad) |
| `A` | Capturar imagen (guarda PNG con timestamp) |

---

## Parametros y Configuracion

### Parametros de vision computacional

| Parametro | Valor | Descripcion |
|:---|:---:|:---|
| `CANNY_LOW_THRESHOLD` | 50 | Umbral bajo de Canny |
| `CANNY_HIGH_THRESHOLD` | 150 | Umbral alto de Canny |
| `GAUSSIAN_KERNEL_SIZE` | (5, 5) | Tamano del kernel Gaussiano |
| `HOUGH_RHO` | 1 | Resolucion en pixeles (Hough) |
| `HOUGH_THETA` | pi/180 | Resolucion angular (Hough) |
| `HOUGH_THRESHOLD` | 15-25 | Votos minimos para linea |
| `HOUGH_MIN_LINE_LENGTH` | 8-20 | Longitud minima de segmento (px) |
| `HOUGH_MAX_LINE_GAP` | 5-30 | Hueco maximo en linea (px) |

### Parametros del controlador

| Parametro | Valor | Descripcion |
|:---|:---:|:---|
| `MAX_STEERING_ANGLE` | 0.25-0.5 | Angulo maximo de direccion (rad) |
| `MAX_STEERING_CHANGE` | 0.025-0.03 | Cambio maximo por paso (rad) |
| `ERROR_SMOOTH_ALPHA` | 0.6 | Factor de suavizado EMA |
| `MIN_ABS_SLOPE` | 0.1 | Pendiente minima para filtrar horizontales |

---

## Resultados y Observaciones

### Hallazgos principales

1. **Umbrales de Canny:** Los valores estandar (80/200) no funcionaron con la resolucion de la camara de Webots. Se redujeron a 50/150 para detectar los bordes del carril.

2. **Control proporcional puro:** A 10 km/h, un controlador con solo ganancia proporcional (`Kp = 0.001`) fue suficiente para seguir el carril de forma estable.

3. **Escalamiento de ganancias:** Al aumentar a 50 km/h, fue necesario reducir `Kp` de 0.001 a 0.0007 para evitar oscilaciones.

4. **Suavizado EMA:** La adicion del promedio movil exponencial en la version final redujo significativamente las oscilaciones causadas por ruido en la deteccion frame a frame.

5. **Termino integral desactivado:** En todas las versiones, `Ki = 0.0` porque el termino integral provocaba oscilaciones fuertes dado que el error lateral cambia rapidamente.

6. **Ganancia derivativa:** Solo en la version final (`Kd = 0.015`) se activo, combinada con el suavizado EMA para que la derivada no amplificara el ruido.

---

## Decisiones de Diseno

| Decision | Justificacion |
|:---|:---|
| Desarrollo incremental (6 pasos) | Permite validar cada etapa del pipeline de forma independiente antes de integrar |
| ROI trapezoidal | Se ajusta a la perspectiva de la camara a bordo, descartando cielo y objetos irrelevantes |
| HoughLinesP sobre HoughLines | La version probabilistica es mas eficiente y retorna segmentos (no lineas infinitas) |
| Filtrado por pendiente | Descarta lineas casi horizontales que no corresponden a marcas del carril |
| Suavizado de direccion (rate limiting) | Previene cambios bruscos que provocan zigzagueo o perdida de control |
| EMA sobre error crudo | Filtra ruido de deteccion sin introducir retardo excesivo |

---

## Trabajo Futuro

- Deteccion de ambos carriles (izquierdo y derecho) para calcular el centro del carril.
- Implementacion de control adaptativo que ajuste las ganancias segun la velocidad.
- Deteccion de curvas mediante ajuste polinomial en lugar de lineas rectas.
- Integracion con sensores adicionales (LiDAR, GPS) para navegacion mas robusta.
- Pruebas con condiciones de iluminacion variable y diferentes escenarios.

---

## Referencias

1. Canny, J. (1986). *A Computational Approach to Edge Detection*. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8(6), 679-698.

2. Duda, R. O., & Hart, P. E. (1972). *Use of the Hough Transformation to Detect Lines and Curves in Pictures*. Communications of the ACM, 15(1), 11-15.

3. OpenCV Documentation. *Canny Edge Detection*. https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html

4. OpenCV Documentation. *Hough Line Transform*. https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html

5. Cyberbotics Ltd. *Webots User Guide*. https://cyberbotics.com/doc/guide/index

6. Ogata, K. (2010). *Modern Control Engineering* (5th ed.). Prentice Hall.

---

## Licencia

Este proyecto esta licenciado bajo la **Licencia Apache 2.0**.
Consultar el archivo [LICENSE](LICENSE) para mas detalles.

---

<div align="center">

*Proyecto academico — Instituto Tecnologico y de Estudios Superiores de Monterrey*
*Maestria en Inteligencia Artificial — Navegacion Autonoma (MR4010.10)*

</div>
