# simple controller with onboard camera - Canny edge detection version
#
# This version keeps the basic Webots vehicle control and adds
# the first image-processing stage required for the activity:
# grayscale conversion + Gaussian blur + Canny edge detection.

from controller import Display, Keyboard
from vehicle import Driver
import numpy as np
import cv2
from datetime import datetime
import os
import time

# Configuration constants
DEBOUNCE_TIME = 0.1  # 100 milliseconds
MAX_ANGLE = 0.5
MAX_SPEED = 250
SPEED_INCR = 5
ANGLE_INCR = 0.05

# Canny configuration constants
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
GAUSSIAN_KERNEL_SIZE = (5, 5)


# Getting image from camera
def get_image(camera):
    """
    Gets the raw image from the Webots camera and converts it to a NumPy array.

    The Webots camera returns the image with 4 channels, so the resulting array
    has the shape: height x width x 4.
    """
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image


def greyscale_cv2(image):
    """
    Converts the camera image to grayscale.

    Grayscale conversion is necessary before applying Canny because Canny works
    on single-channel images.
    """
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img


def canny_edge_detection(gray_image):
    """
    Applies Gaussian blur and then the Canny edge detection algorithm.

    Gaussian blur reduces noise in the image, which helps Canny avoid detecting
    false edges. Canny then highlights strong changes in intensity, which are
    useful for finding lane markings.
    """
    blurred_image = cv2.GaussianBlur(gray_image, GAUSSIAN_KERNEL_SIZE, 0)
    canny_image = cv2.Canny(
        blurred_image,
        CANNY_LOW_THRESHOLD,
        CANNY_HIGH_THRESHOLD
    )
    return canny_image


# Display image on onboard display
def display_image(display, image):
    """
    Displays a single-channel processed image on the onboard Webots display.

    Since the Display device expects an RGB image, the grayscale/Canny image is
    copied into three channels.
    """
    image_rgb = np.dstack((image, image, image))

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
        # 1. Get image from the onboard camera
        image = get_image(camera)

        # 2. Convert the image to grayscale
        grey_image = greyscale_cv2(image)

        # 3. Apply Canny edge detection
        canny_image = canny_edge_detection(grey_image)

        # 4. Display the Canny result on the onboard display
        display_image(display_img, canny_image)

        # To reduce key rebounds
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

        # Print useful debugging information for the video/report
        print(
            "Canny activo | velocidad:",
            speed,
            "| angulo:",
            round(angle, 3),
            "| thresholds:",
            CANNY_LOW_THRESHOLD,
            CANNY_HIGH_THRESHOLD
        )


if __name__ == "__main__":
    main()
