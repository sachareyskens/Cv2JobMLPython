from flask import Flask, request, abort, jsonify
import json
import Classifier_sklearn





app = Flask(__name__)

@app.route('/')
def index():
    return "CV2JOB ML."

@app.route('/scikit/api/predict/<string:predict_string>', methods=['GET'])
def get_prediction(predict_string):
    s = Classifier_sklearn.runPrediction(predict_string, False)
    return s[0]

@app.route('/scikit/api/train', methods=['GET'])
def get_train_accuracy():
    s = Classifier_sklearn.runPrediction("Training", True)
    print(s)
    return s

@app.route('/scikit/api/trainingset', methods=['POST'])
def add_training_data():
    arrayOfTrainings = request.get_json()
    print((arrayOfTrainings))
    return "success"

app.run(debug=True)

