# Actividad 2.1

from controller import Display, Keyboard
from vehicle import Driver
import numpy as np
import cv2
from datetime import datetime
import os
import time

# ============================================================
# CONFIGURACION
# ============================================================
DEBOUNCE_TIME = 0.1     # para que no se repitan las teclas de los controles si se mantiene apretada
MAX_ANGLE = 0.5         # angulo maximo del volante
MAX_SPEED = 250         # vel maxima
SPEED_INCR = 5          # sube/baja la velocidad por cada tecla

# ============================================================
# GANANCIAS DEL PID
# ============================================================
KP = 0.008
KI = 0.0 # Se dejo en 0, provocaba oscilaciones fuertes
KD = 0.015

DEFAULT_STEERING = 0.0  # si el auto no detecta linea, se mueve recto
TARGET_SPEED = 50       # velocidad default del auto

# Suavizado del error:
# 60% del valor viejo y 40% del nuevo
ERROR_SMOOTH_ALPHA = 0.6

# cada 30 pasos imprimo info
DEBUG_EVERY = 30      


# ============================================================
# AUXILIARES
# ============================================================

# webots da la imagen como bytes, se convierte a arreglo de numpy
def get_image(camera):
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )

# paso 3 de la actividad: imagen a escala de grises
def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# mostrar lo mismo que ve el algoritmo en el display de webots
def display_grayscale_with_lines(display, gray_img, lines):
    canvas = gray_img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(canvas, (x1, y1), (x2, y2), 255, 2)
    # el display espera RGB asi que apilo el canal gris 3 veces
    image_rgb = np.dstack((canvas, canvas, canvas,))
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)


# ============================================================
# DETECCION
# ============================================================

# paso 4: bordes con canny.
def canny_edges(gray_img):
    # blur gaussiano porque si no, canny detecta ruido
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    # se usaron los umbrales 50/150, con los valores tipicos (80/200) casi no detectaba nada en una imagen tan chica (128x64)
    return cv2.Canny(blur, 50, 150)

# paso 5: roi
def apply_roi(edges):
    h, w = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(0.00 * w), h),
        (int(0.20 * w), int(0.50 * h)),
        (int(0.80 * w), int(0.50 * h)),
        (int(1.00 * w), h),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)

# paso 6: hough para detectar lineas rectas
def hough_lines(masked_edges):
    return cv2.HoughLinesP(
        masked_edges,
        rho=1, theta=np.pi / 180,
        threshold=15,        # numero de puntos para considerar que hay linea
        minLineLength=8,     # largo minimo de la linea en pixeles
        maxLineGap=5         # cuanto hueco le permito a una linea
    )

# Error
def compute_error(lines, setpoint):
    if lines is None:
        return None
    candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx > 3 * dy:
            continue
        mid_x = (x1 + x2) / 2.0
        candidates.append(mid_x - setpoint)
    if not candidates:
        return None
    return min(candidates, key=abs)


# ============================================================
# MAIN
# ============================================================

def main():
    speed = TARGET_SPEED
    last_press = {}
    prev_error = 0.0
    integral = 0.0
    smoothed_error = None 
    step_count = 0
    no_line_streak = 0
    driver = Driver()
    timestep = int(driver.getBasicTimeStep())

    # camara
    camera = driver.getDevice("camera")
    camera.enable(timestep)
    width = camera.getWidth()
    height = camera.getHeight()

    # setpoint = centro horizontal de la imagen
    # con una camara de 128 de ancho me da setpoint = 64
    SETPOINT = width / 2.0
    print(f"[INFO] Camara {width}x{height}  ->  setpoint = {SETPOINT}")
    display_img = driver.getDevice("display_image")

    # teclado
    keyboard = Keyboard()
    keyboard.enable(timestep)

    print(f"[INFO] PID: KP={KP} KI={KI} KD={KD}  smoothing={ERROR_SMOOTH_ALPHA}")
    print(f"[INFO] velocidad objetivo = {TARGET_SPEED} km/h")

    # bucle principal
    while driver.step() != -1:
        step_count += 1

        # pipeline de vision
        image = get_image(camera)
        grey = to_grayscale(image)
        edges = canny_edges(grey)
        masked = apply_roi(edges)
        lines = hough_lines(masked)
        n_lines = 0 if lines is None else len(lines)

        raw_error = compute_error(lines, SETPOINT)
        if raw_error is not None:
            no_line_streak = 0
            if smoothed_error is None:
                smoothed_error = raw_error
            else:
                smoothed_error = (
                    ERROR_SMOOTH_ALPHA * smoothed_error
                    + (1 - ERROR_SMOOTH_ALPHA) * raw_error
                )
        else:
            no_line_streak += 1
            if no_line_streak > 10:
                smoothed_error = None

        if smoothed_error is not None:
            integral += smoothed_error
            integral = max(-1000.0, min(1000.0, integral))
            derivative = smoothed_error - prev_error
            steering = KP * smoothed_error + KI * integral + KD * derivative
            prev_error = smoothed_error
            steering = max(-MAX_ANGLE, min(MAX_ANGLE, steering))
        else:
            steering = DEFAULT_STEERING
            integral = 0.0
            prev_error = 0.0

        # comandos al carro
        driver.setSteeringAngle(steering)
        driver.setCruisingSpeed(speed)

        # imagen al display
        display_grayscale_with_lines(display_img, grey, lines)

        # debug periodico
        if step_count % DEBUG_EVERY == 0:
            raw_str = f"{raw_error:+.1f}" if raw_error is not None else "None"
            sm_str = f"{smoothed_error:+.1f}" if smoothed_error is not None else "None"
            print(f"[DEBUG] step={step_count}  lineas={n_lines}  "
                  f"raw={raw_str}  suavizado={sm_str}  steering={steering:+.3f}")

        # teclado
        current_time = time.time()
        key = keyboard.getKey()
        # ignorar tecla si se presiono hace poco
        if key in last_press and (current_time - last_press[key] < DEBOUNCE_TIME):
            continue
        last_press[key] = current_time

        # flecha arriba, sube la velocidad
        if key == keyboard.UP:
            if speed < MAX_SPEED:
                speed += SPEED_INCR
                print(f"[KEY] velocidad = {speed}")
        # flecha abajo, baja la velocidad
        elif key == keyboard.DOWN:
            if speed >= SPEED_INCR:
                speed -= SPEED_INCR
                print(f"[KEY] velocidad = {speed}")
        # tecla A, guardo una captura
        elif key == ord('A'):
            stamp = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            file_name = stamp + ".png"
            camera.saveImage(os.getcwd() + "/" + file_name, 1)
            print(f"[KEY] imagen guardada: {file_name}")


if __name__ == "__main__":
    main()