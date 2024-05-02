import cv2

def to_binary(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (900, 900))

    #apply adaptive thresholding to get binary image (101 neighborhood area, 20 constant subtracted from mean)
    binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 20)

    return binary_image
