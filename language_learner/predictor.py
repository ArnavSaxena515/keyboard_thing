# import matplotlib.pyplot as plt
import cv2
import keras
import numpy as np
import random
# from keras.models import Sequential
# from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
# from tensorflow.keras.optimizers import SGD, Adam
# from keras.callbacks import ReduceLROnPlateau, EarlyStopping
# from tensorflow.keras.utils import to_categorical
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
from tkinter.filedialog import askopenfilename


def image_processing(img):
    img_copy = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400, 440))
    img_copy = cv2.GaussianBlur(img_copy, (7, 7), 0)
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

    img_final = cv2.resize(img_thresh, (28, 28))
    img_final = np.reshape(img_final, (1, 28, 28, 1))
    return img_final


word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
             13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
             25: 'Z'}
model = keras.models.load_model(
    r'C:\Users\arnav\PycharmProjects\CNN character recognition\model_hand.h5')


def predict_letter(img_current, char_to_check):
    # pathFullImage = askopenfilename(initialdir="/", title="Select a File", filetypes=[
    #
    #     ('image files', '.png'),
    #     ('image files', '.jpg'),
    # ])
    # img_current = cv2.imread(pathFullImage)
    img_final = image_processing(img_current)
    img_pred = word_dict[np.argmax(model.predict(img_final))]
    score = (max(max(model.predict(img_final))))



    print(f"Score: {score}")
    print("Predicted letter: " + char_to_check)

    return char_to_check, score
    # cv2.putText(img_current, "Prediction _ _ _ ", (20, 25), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color=(0, 0, 230))
    # cv2.putText(img_current, "Prediction: " + img_pred, (20, 410), cv2.FONT_HERSHEY_DUPLEX, 1.3, color=(255, 0, 30))
    # cv2.imshow('Character recognition _ _ _ ', img_current)
    # while (1):
    #     k = cv2.waitKey(1) & 0xFF
    #     if k == 27:
    #         break
    # cv2.destroyAllWindows()
