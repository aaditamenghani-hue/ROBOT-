# --- IMPORT LIBRARIES ---
# cv2 and numpy for computer vision
# gpiozero for controlling the servo motor
# pygame for playing audio
# os and random for file management and random choice
import cv2
import numpy as np
from gpiozero import Servo
from time import sleep
import pygame
import os
import random


# --- SERVO SETUP ---
# The servo is connected to GPIO pin 18 on your Raspberry Pi
# The servo will start in the closed position
servo = Servo(18)
servo.value = -1  # -1 = fully closed, 1 = fully open

# --- CAMERA SETUP ---
# Open the first connected camera (index 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()


# --- AUDIO SETUP ---
pygame.mixer.init()

# Folder that stores your MP3 files
# Make sure this folder path is correct
audio_folder = "/home/andrewstapleton/Downloads/audio files"

# Audio file paths
audio_triangle = os.path.join(audio_folder, "give_me_triangle.mp3")
audio_square = os.path.join(audio_folder, "give_me_square.mp3")
audio_good = os.path.join(audio_folder, "good_job.mp3")
audio_uh = os.path.join(audio_folder, "uh_oh.mp3")


# --- AUDIO PLAY FUNCTION ---
# Plays an audio file if it exists, otherwise prints a warning
def play_audio(file):
    if not os.path.exists(file):
        print(f"⚠️ Audio file not found: {file}")
        return
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()
    # Wait until the audio finishes before continuing
    while pygame.mixer.music.get_busy():
        sleep(0.1)


# --- SERVO MOVEMENT HELPERS ---
# Smoothly moves the servo between positions instead of jumping suddenly
def smooth_move(target_value, steps=40, delay=0.05):
    current = servo.value if servo.value is not None else -1
    for v in np.linspace(current, target_value, steps):
        servo.value = v
        sleep(delay)

# Opens the servo "mouth"
def open_mouth():
    smooth_move(1)
    print("Mouth OPEN")

# Closes the servo "mouth"
def close_mouth():
    smooth_move(-1)
    print("Mouth CLOSED")


# --- SHAPE DETECTION FUNCTION ---
# Detects whether a triangle or square appears in the camera frame
def detect_shapes(frame):
    # Convert to grayscale to simplify the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Blur the image to reduce noise and make edges smoother so then it can be detected better 
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold turns the image black & white for easier contour detection and avoid confusion
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the contours (edges/outlines) of shapes in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_shape = None

    # Loop through all found contours
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1500:
            # Skip small contours that are just noise
            continue

        # Approximate the contour to a polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        sides = len(approx)

        # If the polygon has 3 sides → triangle
        if sides == 3:
            detected_shape = "triangle"
            cv2.drawContours(frame, [approx], -1, (0, 255, 255), 3)

        # If it has 4 sides → could be square
        elif sides == 4:
            # Get the bounding box and check aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.75 <= aspect_ratio <= 1.25:
                detected_shape = "square"
                cv2.drawContours(frame, [approx], -1, (255, 0, 0), 3)

    return detected_shape, frame


# --- MAIN GAME LOOP ---
try:
    while True:
        # Randomly choose whether Dino wants a triangle or square
        target_shape = random.choice(["triangle", "square"])

        # Play the corresponding audio file
        if target_shape == "triangle":
            play_audio(audio_triangle)
        else:
            play_audio(audio_square)

        print(f"🗣️ Dino says: Give me a {target_shape.upper()}!")

        shape_found = False

        # Keep looping until the correct shape is found
        while not shape_found:
            ret, frame = cap.read()
            if not ret:
                print("Error: Frame not captured.")
                break

            # Detect shapes in the camera frame
            detected_shape, frame = detect_shapes(frame)

            # If the correct shape is shown
            if detected_shape == target_shape:
                print(f" Correct! You showed a {detected_shape.upper()}")
                open_mouth()
                play_audio(audio_good)
                shape_found = True

            # If the wrong shape is shown
            elif detected_shape is not None:
                print(f" Wrong shape ({detected_shape}), wanted {target_shape}")
                close_mouth()
                play_audio(audio_uh)

            # Display the camera feed with shape outlines
            cv2.imshow("Camera", frame)

            # Press 'q' to quit the program safely
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

            # Slow down detection so it’s not too jumpy
            sleep(0.5)

        # Close the mouth and wait before next round
        close_mouth()
        sleep(2.0)

# --- SAFETY EXIT ---
except KeyboardInterrupt:
    print(" Stopping program...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    close_mouth()
    print("Program exited safely.")

