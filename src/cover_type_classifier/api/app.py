from crypt import methods
import yaml
import pickle
import pandas as pd
from flask import Flask
from flask import request, jsonify

app = Flask(__name__)

# get configuration
config_path = "config.yml"

with open(config_path, "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

with open(config["api_model_path"], "rb") as f_in:
    model, pipeline = pickle.load(f_in)


@app.route("/predict", methods=["POST"])
def predict():
    print(request.get_json(force=True))
    data = pd.json_normalize(request.get_json())
    data.columns = data.columns.str.lower()
    data_transformed = pipeline.transform(data)
    prediction = model.predict(data_transformed)

    result = {"forest_classes": int(prediction)}
    return jsonify(result)
    return 'Sent'


@app.route("/get_data", methods=["GET"])
def get_data():
    return "My life sucks"


def main():
    app.run(port=5432)


if __name__ == "__main__":
    main()
