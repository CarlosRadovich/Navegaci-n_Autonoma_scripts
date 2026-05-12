# simple controller with onboard camera - Hough transform version
#
# This version keeps the basic Webots vehicle control and adds:
# 1. Grayscale conversion
# 2. Gaussian blur
# 3. Canny edge detection
# 4. Region of interest using fillPoly
# 5. Line detection using HoughLinesP
#
# The goal of this stage is to verify that the program can detect straight
# line segments from the camera image before adding the PID controller.

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

# Hough transform configuration constants
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 25
HOUGH_MIN_LINE_LENGTH = 20
HOUGH_MAX_LINE_GAP = 30


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


def region_of_interest(edge_image):
    """
    Defines and applies a polygonal region of interest.

    The purpose of this mask is to keep only the lower part of the image, where
    the lane markings are expected to appear from the vehicle camera.
    Areas outside the polygon are ignored.
    """
    height, width = edge_image.shape

    # Create a black mask with the same size as the Canny image.
    mask = np.zeros_like(edge_image)

    # Trapezoidal region of interest.
    # These values can be adjusted depending on the camera position.
    polygon = np.array([[
        (int(0.05 * width), height),
        (int(0.95 * width), height),
        (int(0.65 * width), int(0.55 * height)),
        (int(0.35 * width), int(0.55 * height))
    ]], dtype=np.int32)

    # Fill the polygon in white.
    cv2.fillPoly(mask, polygon, 255)

    # Keep only the pixels inside the region of interest.
    masked_image = cv2.bitwise_and(edge_image, mask)

    return masked_image


def detect_hough_lines(masked_edges):
    """
    Detects straight line segments using the probabilistic Hough transform.

    HoughLinesP returns a list of detected line segments. Each line contains
    four values: x1, y1, x2, y2.
    """
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )

    return lines


def draw_hough_lines(base_image, lines):
    """
    Draws the detected Hough lines over a blank image.

    The input image from the camera has 4 channels. For visualization, this
    function creates a 3-channel image and draws the detected lines in white.
    """
    height = base_image.shape[0]
    width = base_image.shape[1]

    line_image = np.zeros((height, width, 3), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    return line_image


# Display image on onboard display
def display_image(display, image):
    """
    Displays a processed image on the onboard Webots display.

    If the image has one channel, it is copied into three channels.
    If it already has three channels, it is displayed directly.
    """
    if len(image.shape) == 2:
        image_rgb = np.dstack((image, image, image))
    else:
        image_rgb = image

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

        # 4. Apply the region of interest
        roi_image = region_of_interest(canny_image)

        # 5. Detect straight lines with HoughLinesP
        lines = detect_hough_lines(roi_image)

        # 6. Draw the detected lines for visualization
        hough_image = draw_hough_lines(image, lines)

        # 7. Display the Hough result on the onboard display
        display_image(display_img, hough_image)

        # Count detected lines
        number_of_lines = 0 if lines is None else len(lines)

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
            "Hough activo | velocidad:",
            speed,
            "| angulo:",
            round(angle, 3),
            "| lineas detectadas:",
            number_of_lines
        )


if __name__ == "__main__":
    main()
