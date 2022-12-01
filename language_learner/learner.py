import random

import cv2
import numpy as np
import mediapipe as mp
import ctypes
# from cvzone.HandTrackingModule import HandDetector as htm
import cvzone.HandTrackingModule as htm
from tkinter import *
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

from language_learner.predictor import predict_letter

MB_OK = 0
MB_OKCANCEL = 1
MB_YESNOCANCEL = 3
MB_YESNO = 4

IDOK = 1
IDCANCEL = 2
IDABORT = 3
IDYES = 6
IDNO = 7


def back():
    print("Back Pressed")


def run_learner():
    win = Tk()
    win.geometry("1600x1000")
    characters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                  "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
                  "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    text = "This is a letter"
    coordinates = (30, 35)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 255)
    thickness = 2
    headerImage = cv2.imread("Canvas.png")
    pencilImage = cv2.imread("pencil_icon.png")
    # print(len(pencilImage))
    # print(len(pencilImage[0]))
    # print(np.shape(pencilImage))

    # Capture the video camera
    video_feed = cv2.VideoCapture(0)
    # Set dimensions for webcam window
    video_feed.set(3, 1280)
    video_feed.set(4, 720)

    brush_thickness = 15
    eraser_thickness = 50
    draw_color = (255, 0, 255)
    red_color = (255, 0, 0)
    blue_color = (0, 0, 255)
    pink_color = (255, 0, 255)

    hand_detect = htm.HandFinder(detection_confidence=0.85, number_of_hands=1)
    x_prev, y_prev = 0, 0
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
    mpDraw = mp.solutions.drawing_utils
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    char_to_draw = random.choice(characters)

    while True:
        # imgCanvas = cv2.putText(imgCanvas, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
        # cv2.namedWindow("Frame")
        # cv2.createButton("Back", back, None, cv2.QT_PUSH_BUTTON, 1)

        # 1. Import image
        success, live_image = video_feed.read()

        imgCanvas = cv2.putText(imgCanvas, char_to_draw, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)

        live_image = cv2.flip(live_image, 1)  # Flip the image to solve the mirror issue
        # live_image[0:128, 0:1280] = headerImage

        # 2 Find Hand Landmarks
        live_image = hand_detect.findHands(live_image)
        hand_landmarks = hand_detect.findPosition(live_image, draw=False)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(imgCanvas))

        if len(hand_landmarks) != 0:
            # print(landmarks)
            # Tip of index and middle fingers
            x1, y1 = hand_landmarks[8][1:]  # index finger tip
            x2, y2 = hand_landmarks[12][1:]  # middle finger tip
            transform_image = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2RGB)
            hand_info = hands.process(transform_image)

            # np.add(imgCanvas[x1:x1 + 48, y1:y1 + 48], pencilImage)
            # T = np.float32([[1, 0, x1-x_prev], [0, 1, y1-y_prev]])

            # 3 Check which fingers are up
            fingers_active = hand_detect.active_fingers()

            # 4 If selection mode - Two fingers are up
            if fingers_active[1] and fingers_active[2]:
                cv2.rectangle(live_image, (x1, y1 - 15), (x2, y2 + 15), draw_color, cv2.FILLED)
                if y1 < 128:  # value of header
                    if 15 < x1 < 215:
                        draw_color = (0, 0, 255)
                        print("RED")
                        print(draw_color)
                    elif 300 < x1 < 500:
                        draw_color = (255, 0, 0)
                        print("BLUE")
                        print(draw_color)
                    elif 630 < x1 < 830:
                        draw_color = pink_color
                        print("PINK")
                        print(draw_color)
                    elif 1115 < x1 < 1260:
                        draw_color = (0, 0, 0)
                        print("ERASER")
                # np.subtract(imgCanvas[x_prev:x_prev + 48, y_prev + 48], pencilImage)
                x_prev, y_prev = x1, y1

            if fingers_active[0] and fingers_active[1] and fingers_active[2] and fingers_active[3] and fingers_active[
                4]:
                imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                # imgCanvas = cv2.putText(imgCanvas, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)

            if fingers_active[4] and fingers_active[1] and fingers_active[2] and fingers_active[3]:
                result = ctypes.windll.user32.MessageBoxW(0, "Submit drawing?", "Submit triggered", 3)
                if result == IDYES:
                    print("Predicting")
                    predicted_letter, score = predict_letter(imgCanvas, char_to_draw)
                    ctypes.windll.user32.MessageBoxW(0, f"Predicted Letter:{predicted_letter}\nYour score: {score}",
                                                     "Results", 0)
                    char_to_draw = random.choice(characters)
                    continue

                else:
                    continue
                # print(predicted_letter)
                # print(score)
                # imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                # imgCanvas = cv2.putText(imgCanvas, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)





            # 5 If drawing mode - Index finger is up
            elif fingers_active[1] and fingers_active[2] == False:
                # np.subtract(imgCanvas[x_prev:x_prev + 48, y_prev:y_prev + 48], pencilImage)
                cv2.circle(live_image, (x1, y1), 15, draw_color, cv2.FILLED)
                if x_prev == 0 and y_prev == 0:  # first frame
                    x_prev, y_prev = x1, y1
                if draw_color == (0, 0, 0):
                    cv2.line(live_image, (x_prev, y_prev), (x1, y1), draw_color, thickness=eraser_thickness)
                    cv2.line(imgCanvas, (x_prev, y_prev), (x1, y1), draw_color, thickness=eraser_thickness)

                # cv2.line(img, (xp, yp), (x1, y1), drawColor, thickness=brushThickness)
                cv2.line(imgCanvas, (x_prev, y_prev), (x1, y1), draw_color, thickness=brush_thickness)
                x_prev, y_prev = x1, y1
            else:
                x_prev, y_prev = x1, y1

        gray_image = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, image_inverse = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
        image_inverse = cv2.cvtColor(image_inverse, cv2.COLOR_GRAY2BGR)
        live_image = cv2.bitwise_and(live_image, image_inverse)
        live_image = cv2.bitwise_or(live_image, imgCanvas)
        cv2.imshow("Image", live_image)
        cv2.imshow("Image Canvas", imgCanvas)

        key = cv2.waitKey(1)
        if key == ord('q'):
            video_feed.release()
            cv2.destroyAllWindows()
            break

    Label(win, image=imgtk).pack()
    win.mainloop()
    return
    #
