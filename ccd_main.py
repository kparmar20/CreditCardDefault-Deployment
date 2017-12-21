import sys
import os
import shutil
import traceback
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.externals import joblib

from utils import model_utils

app = Flask(__name__)

model = None


@app.route('/test_endpoint', methods=['GET'])
def test_function():
    print("I made my own endpoint!")
    return "test endpoint made ccd_main.py"


@app.route('/predict', methods=['POST'])
def predict():
    #print("Data is\n", request.json)
    if model:
        print("model exists")
        try:
            input_df = pd.DataFrame(request.json)
            predictions = model_utils.predict(input_df, model)
            return jsonify(predictions)
        except Exception as e:
            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('You need to train a model before you can make predictions.')
        return 'error: no model'


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 5000

    try:
        model = joblib.load(model_utils.MODEL_FILE_NAME)
        print('model loaded')
    except Exception as e:
        print('No model here')
        print('Train first')
        print(str(e))
        model = None

    app.run(host='0.0.0.0', port=port, debug=True)
