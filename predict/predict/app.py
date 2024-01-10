from flask import Flask, request
from predict.predict import run as run_predict
from train.train import run as run_train
import json

app = Flask(__name__)


@app.route('/predict', methods=["GET"])
def predict():
    artefact_path = "C:/1_QUENTIN/EPF/TAFF/5A/From PoC to Prod/poc-to-prod-capstone/train/data/artefacts/2024-01-09-11-30-55"
    list_text = ['Why am I unable to install Tensorflow?']
    model = run_predict.TextPredictionModel.from_artefacts(artefact_path)
    prediction = model.predict(list_text, top_k=5)
    names = [model.labels_to_index[str(idx)] for idx in prediction[0]]
    result_json = json.dumps(names)
    return result_json


if __name__ == '__main__':
    app.run(debug=True)
