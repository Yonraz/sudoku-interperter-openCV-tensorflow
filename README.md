# OpenCV  Sudoku Interpreter

## Python | OpenCV | Tensorflow | Flask

### Extract sudoku board data from an image and return a grid

## Overview
This api is made for a sudoku solver client, which allows users to upload images of a sudoku board and solve it for them.
This app is focused on the image processing part, it simply receives a file, processes it and returns a matrix.
I did not have much prior knowledge in machine learning or image processing, and this solution is based on multiple documentations, articles and tutorials.  

##  Machine Learning
To handle digit recognition, I first had to create an OCR (optical character recognition).
I used the mnist dataset for training the model. Mnist is a large data set which consists of handwritten digits. Currently the model is only trained on mnist, and its final iteration can be seen on data_augmentation.py. I will need to add in digital fonts to make it more reliable for online sudoku screenshots.
Data augmentation was used to enhance the model's ability to detect digits in different sizes, rotations, and skewed perspective.

## Image Processing
To handle processing, several steps are needed:
* __Find The Board__:  We need to find the area of the sudoku board, in order to crop it and use it separately from the image. ('warp_persp.py')
this is done by first applying some preprocessing (blurring and adaptive threshold) which will make it easier to find the borders of the board - the blur effect reduces the amount of contours and adaptive thresholding turns the image into a binary image, making each value either 0 or 255 depending on a specified threshold.

- __Warp Perspective__ : Then we need to realign the found corners of the board in order to normalize the image (for images with skewed perspectives). 
One strategy to achieve this is by calculating the Euclidean distance between the upper width and lower width, and distance between the left height and right height, then taking the max value from each and getting our desired width and height.
then, with the max width and height we create a transformation matrix:
```python 
		dimensions = np.array([[0, 0], [width, 0], [width, width],
				          [0, width]], dtype="float32")
```
and apply it to our image.

- __Convert To Binary Image__: now that we have the image cropped and aligned, we need to add one more process before we can chop it into cells of digits:
The image must be converted to binary (black and white) to help the tf model achieve better predictions, since the dataset it's trained on is also binary. Also we resize the image to 900x900, which will make it easier for cropping.
```python
	binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 20)
```
- __Break into Cells__: Now our image is ready. we need to crop the image into 81 different cells and place the final predictions in a grid.
```python
	for row in range(1, 10):
            for col in range(1, 10):
                cell = binary_img[row*100-100+crop_amt:row*100-crop_amt, col*100-100+crop_amt:col*100-crop_amt]
                cell = cv2.resize(cell, (28, 28))
                res = get_prediction(cell, model)
                grid[row-1][col-1] = res
```
we take 81 iterations (1 through 10) and crop out pixels from the image in 100x100 pieces (from width(0,100) height(0,100) and so on)
each cell is resized to 28x28 (the mnist dataset's training data size).
then a prediction is made from the cell's image data. if the prediction confidence is less than 0.9 it probably means we hit an empty cell which wasn't included in the data set, so it's defaulted to 0.

## Web Server
The web server aspect of the project was the least important part for me, it is written in flask and simply receives  one post request with file data, tries to process it and returns a finished grid. error handling hasn't really been done except for the simplest cases.
				      
