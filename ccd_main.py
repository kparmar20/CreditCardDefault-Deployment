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
    print("Data is\n", request.json)
    input_df=model_utils.transform(request.json)
    print("Data transformed\n",input_df)

    if model:
        print("model exists")
        try:
            #input_df = pd.DataFrame(request.json)
            predictions = model_utils.predict(input_df, model)
            #print("Predictions", predictions)
            return jsonify(predictions)
        except Exception as e:
            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('You need to train a model before you can make predictions.')
        return 'error: no model'


@app.route('/update', methods=['POST'])
def update():
    #print("Data is\n", request.json)
    X=model_utils.transform_update(request.json)[0]
    y=model_utils.transform_update(request.json)[1]
    #print("X ",X)
    #print("y ",y)
    model_utils.update(model,X,y)
    joblib.dump(model, model_utils.MODEL_FILE_NAME)
    print("new model dumped")    

    return "Success"


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
