from flask import Flask, jsonify, request
from server.process_img import get_grid_from_sudoku
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)
UPLOAD_PATH='server'
FILE_NAME='img'

def getFileName(name):
    return FILE_NAME + os.path.splitext(name)[1]

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                return jsonify('No file part')
        file = request.files['file']
        if file.filename == '':
            return jsonify('No selected file')
        if file:
            filename = os.path.join(UPLOAD_PATH, getFileName(file.filename))
            file.save(filename)
            grid = get_grid_from_sudoku(filename)
            os.remove(filename)
            return jsonify(grid)
    except Exception as e:
        print(e)
        return jsonify("Error")

if __name__ == '__main__':
    app.run(debug=True)
