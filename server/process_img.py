from json import JSONEncoder
import json
from warp_persp import get_square_box_from_image
from cvt_to_binary import to_binary
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from predict import get_prediction
import os

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def get_grid_from_sudoku(filename):
    try:
        original = cv2.imread(filename)

        sudoku = get_square_box_from_image(original) # align perspective and crop to square

        binary_img = to_binary(sudoku) # convert to 900x900 binary image

        crop_amt = 20 # amount to crop from each side to avoid the border

        grid = np.zeros((9, 9), dtype=int) # 9x9 grid to store the final result

        model = tf.keras.models.load_model('server/mnist_da_model.keras') # load model


        for row in range(1, 10):
            for col in range(1, 10):
                print(row, col)
                cell = binary_img[row*100-100+crop_amt:row*100-crop_amt, col*100-100+crop_amt:col*100-crop_amt]
                cell = cv2.resize(cell, (28, 28))

                res = get_prediction(cell, model)
                grid[row-1][col-1] = res
                
        numpy_data = {
            "grid": grid
        }
        json_data = json.dumps(numpy_data, cls=NumpyArrayEncoder)
        print(json_data)
        return json_data
    except Exception as e:
        print(e)
        return None