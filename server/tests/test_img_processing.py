from warp_persp import get_square_box_from_image
from cvt_to_binary import to_binary
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from test_mnist import get_prediction

original = cv2.imread("server/sudoku_bg.png")


sudoku = get_square_box_from_image(original) # align perspective and crop to square

binary_img = to_binary(sudoku) # convert to 900x900 binary image

crop_amt = 20 # amount to crop from each side to avoid the border

grid = np.zeros((9, 9), dtype=int) # 9x9 grid to store the final result

model = tf.keras.models.load_model('mnist_da_model.keras') # load model


for row in range(1, 10):
    for col in range(1, 10):
        cell = binary_img[row*100-100+crop_amt:row*100-crop_amt, col*100-100+crop_amt:col*100-crop_amt]
        cell = cv2.resize(cell, (28, 28))

        # cv2.imshow(f"Window {row}_{col}",binary_img[row*100-100+crop_amt:row*100-crop_amt, col*100-100+crop_amt:col*100-crop_amt])
        res = get_prediction(cell, model)
        grid[row-1][col-1] = res

# fig, ax = plt.subplots()
# for i in range(len(grid)):
#     for j in range(len(grid[0])):
#         ax.text(j, i, str(grid[::-1][i][j]), va='center', ha='center')

# ax.axis('off')  # Turn off the axis
# plt.show()

print(grid)