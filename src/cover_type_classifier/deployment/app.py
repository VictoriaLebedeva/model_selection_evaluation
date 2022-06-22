import click
from src.cover_type_classifier.models.predict import predict
from flask import Flask, request, jsonify

# initialize app
app = Flask(__name__)

cover_types = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}


@app.route("/predict", methods=["POST"])
def predict_value():
    result = []
    data = request.get_json()
    for row in data:
        prediction = predict(app.config["model_path"], row)
        output = {
            "code": prediction.tolist(),
            "cover_type": cover_types[prediction[0]],
        }
        result.append(output)
    return jsonify(result)


@click.command()
@click.option(
    "--model-path",
    default="models\\models\\random_forest_18052022_211019.bin",
    type=click.Path(exists=True, dir_okay=False),
    help="Patn to the model.",
)
def main(model_path: str) -> None:
    app.config["model_path"] = model_path
    app.run(debug=False, host="0.0.0.0", port=9696)


if __name__ == "__main__":
    main()
