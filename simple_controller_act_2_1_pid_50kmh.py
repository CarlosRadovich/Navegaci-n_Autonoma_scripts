# simple controller with onboard camera - 50 km/h PID test version
#
# This file is intended for the final activity test at 50 km/h.
# It uses the same image-processing pipeline:
# camera -> grayscale -> Canny -> ROI -> HoughLinesP -> error -> PID/steering.
#
# The controller is intentionally smoothed to reduce sudden steering changes.

from controller import Display
from vehicle import Driver
import numpy as np
import cv2
import time

# ============================================================
# General configuration
# ============================================================

# Activity requested minimum speed validation: 50 km/h.
CRUISING_SPEED = 50

# Steering limits.
MAX_STEERING_ANGLE = 0.25
MAX_STEERING_CHANGE = 0.025
DEFAULT_STEERING_ANGLE = 0.0

# ============================================================
# Image processing configuration
# ============================================================

CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
GAUSSIAN_KERNEL_SIZE = (5, 5)

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 25
HOUGH_MIN_LINE_LENGTH = 20
HOUGH_MAX_LINE_GAP = 30

# Lower value allows more detected lines to be used.
MIN_ABS_SLOPE = 0.1

# ============================================================
# PID gains
# ============================================================
# Conservative gains for smoother behavior at 50 km/h.
KP = 0.0007
KI = 0.0
KD = 0.0


class PIDController:
    """
    Basic PID controller.

    For this 50 km/h test, Ki and Kd are set to zero, so the behavior is
    mostly proportional and less aggressive.
    """

    def __init__(self, kp, ki, kd, output_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit

        self.previous_error = 0.0
        self.integral = 0.0
        self.previous_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.previous_time

        if dt <= 0:
            dt = 1e-6

        self.integral += error * dt
        derivative = (error - self.previous_error) / dt

        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        output = max(-self.output_limit, min(self.output_limit, output))

        self.previous_error = error
        self.previous_time = current_time

        return output


def limit_steering_change(current_angle, target_angle):
    """
    Limits how much the steering angle can change in one simulation step.
    """
    angle_change = target_angle - current_angle
    angle_change = max(
        -MAX_STEERING_CHANGE,
        min(MAX_STEERING_CHANGE, angle_change)
    )

    new_angle = current_angle + angle_change
    new_angle = max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, new_angle))

    return new_angle


def get_image(camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image


def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img


def canny_edge_detection(gray_image):
    blurred_image = cv2.GaussianBlur(gray_image, GAUSSIAN_KERNEL_SIZE, 0)
    canny_image = cv2.Canny(
        blurred_image,
        CANNY_LOW_THRESHOLD,
        CANNY_HIGH_THRESHOLD
    )
    return canny_image


def region_of_interest(edge_image):
    height, width = edge_image.shape

    mask = np.zeros_like(edge_image)

    polygon = np.array([[
        (int(0.05 * width), height),
        (int(0.95 * width), height),
        (int(0.65 * width), int(0.55 * height)),
        (int(0.35 * width), int(0.55 * height))
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(edge_image, mask)

    return masked_image


def detect_hough_lines(masked_edges):
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )

    return lines


def calculate_lane_error(lines, image_width):
    """
    Calculates the error used by the controller.

    The setpoint is the horizontal center of the image.
    The selected error is the smallest error from the valid detected lines.
    """
    setpoint = image_width / 2
    best_error = None

    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line[0]

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            slope = float("inf")
        else:
            slope = dy / dx

        if abs(slope) < MIN_ABS_SLOPE:
            continue

        line_mid_x = (x1 + x2) / 2
        error = setpoint - line_mid_x

        if best_error is None or abs(error) < abs(best_error):
            best_error = error

    return best_error


def draw_hough_lines(base_image, lines):
    height = base_image.shape[0]
    width = base_image.shape[1]

    line_image = np.zeros((height, width, 3), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    return line_image


def display_image(display, image):
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


def main():
    driver = Driver()

    timestep = int(driver.getBasicTimeStep())

    camera = driver.getDevice("camera")
    camera.enable(timestep)

    display_img = Display("display_image")

    controller = PIDController(
        kp=KP,
        ki=KI,
        kd=KD,
        output_limit=MAX_STEERING_ANGLE
    )

    driver.setCruisingSpeed(CRUISING_SPEED)

    steering_angle = DEFAULT_STEERING_ANGLE

    while driver.step() != -1:
        image = get_image(camera)

        grey_image = greyscale_cv2(image)
        canny_image = canny_edge_detection(grey_image)
        roi_image = region_of_interest(canny_image)
        lines = detect_hough_lines(roi_image)

        image_width = camera.getWidth()
        error = calculate_lane_error(lines, image_width)

        if error is not None:
            target_angle = controller.update(error)

            # If the vehicle turns in the wrong direction, replace with:
            # target_angle = -controller.update(error)
        else:
            target_angle = steering_angle * 0.8

        steering_angle = limit_steering_change(steering_angle, target_angle)

        driver.setSteeringAngle(steering_angle)
        driver.setCruisingSpeed(CRUISING_SPEED)

        hough_image = draw_hough_lines(image, lines)
        display_image(display_img, hough_image)

        number_of_lines = 0 if lines is None else len(lines)

        print(
            "PID 50 km/h | velocidad:",
            CRUISING_SPEED,
            "| lineas:",
            number_of_lines,
            "| error:",
            error,
            "| target:",
            round(target_angle, 4),
            "| angulo:",
            round(steering_angle, 4)
        )


if __name__ == "__main__":
    main()
