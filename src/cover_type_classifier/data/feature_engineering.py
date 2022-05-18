import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from cover_type_classifier.data.custom_estimator import (
    RemoveIrrelevantFeatures,
    NoTransformation,
)


def transformation_pipeline(
    remove_irrelevant_features_flag: bool, min_max_scaler: bool
) -> Pipeline:
    pipeline_steps = []
    if min_max_scaler:
        pipeline_steps.append(("min_max_scaler", MinMaxScaler()))
    if remove_irrelevant_features_flag:
        pipeline_steps.append(
            ("remove_irrelevant_features", RemoveIrrelevantFeatures())
        )

    if len(pipeline_steps) > 0:
        return Pipeline(steps=pipeline_steps)
    else:
        return Pipeline(steps=[("do nothing", NoTransformation())])


if __name__ == "__main__":
    df = pd.read_csv("data/external/train.csv")
    target = df["Cover_Type"]
    features = df.drop("Cover_Type", axis=1)
    features_test = pd.read_csv("data/external/test.csv")

    pipeline = transformation_pipeline(True, True)
    pipeline.fit(features, target)
    pipeline.transform(features_test)
