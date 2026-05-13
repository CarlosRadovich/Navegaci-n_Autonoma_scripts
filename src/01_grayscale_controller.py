"""
===============================================================================
  Actividad 2.1 — Navegacion Autonoma (MR4010.10)
  Paso 1: Controlador basico con conversion a escala de grises
===============================================================================

  Descripcion:
      Controlador base para el vehiculo en Webots. Captura la imagen de la
      camara a bordo, la convierte a escala de grises y la muestra en el
      display integrado. El vehiculo se controla manualmente con el teclado.

  Pipeline:  Camara -> Escala de grises -> Display

  Equipo:
      Antonio Olvera Donlucas          A01795617
      Carlos Monir Radovich Saad       A01797569
      Andres Roberto Osuna Gonzalez    A01796264
      Oscar Alberto Ramirez Anaya      A01795438

  Institucion:
      Instituto Tecnologico y de Estudios Superiores de Monterrey
      Maestria en Inteligencia Artificial

  Fecha: Mayo 2026
===============================================================================
"""
# simple controller with onboard camera

from controller import Display, Keyboard, Camera
from vehicle import Driver
import numpy as np
import cv2
from datetime import datetime
import os
import time

# configuration constants
DEBOUNCE_TIME = 0.1  # 100 milliseconds
MAX_ANGLE = 0.5
MAX_SPEED = 250
SPEED_INCR = 5
ANGLE_INCR = 0.05

# Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

# Image processing example
def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

# Display image on onboard display
def display_image(display, image):
    # Image to display
    image_rgb = np.dstack((image, image, image))

    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

# main
def main():
    speed = 10
    angle = 0.0
    last_press = {}

    # Create the Driver instance.
    # Driver is enough to control the car and access devices such as the camera.
    driver = Driver()

    # Get the time step of the current world.
    timestep = int(driver.getBasicTimeStep())

    # Create camera instance
    camera = driver.getDevice("camera")
    camera.enable(timestep)

    # Processing display
    display_img = Display("display_image")

    # Create keyboard instance
    keyboard = Keyboard()
    keyboard.enable(timestep)

    while driver.step() != -1:
        # Get image from camera
        image = get_image(camera)

        # Process and display image
        grey_image = greyscale_cv2(image)
        display_image(display_img, grey_image)

        # To reduce rebounds
        current_time = time.time()

        # Read keyboard
        key = keyboard.getKey()

        if key in last_press and (current_time - last_press[key] < DEBOUNCE_TIME):
            continue  # Ignore rebound

        # Pressed key accepted, update
        last_press[key] = current_time

        if key == keyboard.UP:  # up
            if speed < MAX_SPEED:
                speed += SPEED_INCR
                print("up")
        elif key == keyboard.DOWN:  # down
            if speed >= SPEED_INCR:
                speed -= SPEED_INCR
                print("down")
        elif key == keyboard.RIGHT:  # right
            angle += ANGLE_INCR
            if angle > MAX_ANGLE:
                angle = MAX_ANGLE
            print("right")
        elif key == keyboard.LEFT:  # left
            angle -= ANGLE_INCR
            if angle < -MAX_ANGLE:
                angle = -MAX_ANGLE
            print("left")
        elif key == ord('A'):
            # Filename with timestamp and saved in current directory
            current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            file_name = current_datetime + ".png"
            print("Image taken")
            camera.saveImage(os.getcwd() + "/" + file_name, 1)

        # Update angle and speed
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()
