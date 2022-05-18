import click
import pickle


def load_model(model_path: str) -> object:
    """Loads model from binary file."""
    with open(model_path, "rb") as file:
        model, pipeline = pickle.load(file)
    return model


@click.command()
@click.option(
    "--model-path",
    "-np",
    type=click.Path(exists=True, dir_okay=False),
)
def predict(model_path, data):
    model, pipeline = load_model(model_path)
    data = pipeline.transform(data)
    prediction = model.predict(data)
    return prediction
