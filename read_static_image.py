import cv2
import numpy as np

# preprocess image
img = cv2.imread('sudoku.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
canny = cv2.Canny(blur,50,50)
img_copy = img.copy()

contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# get outer bounds of board
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 60000:
        cv2.drawContours(img_copy, [cnt], -1, (0,255,0), 2)

