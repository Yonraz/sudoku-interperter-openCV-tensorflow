from flask import Flask, jsonify, request
from process_img import get_grid_from_sudoku
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)
UPLOAD_PATH='server'
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
            filename = os.path.join(UPLOAD_PATH, file.filename)
            file.save(filename)
            grid = get_grid_from_sudoku(filename)
            os.remove(filename)
            return jsonify(grid)
    except Exception as e:
        print(e)
        return jsonify("Error")

if __name__ == '__main__':
    app.run(debug=True)
