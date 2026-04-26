import pytest
import pandas as pd
from src.pipelines.prediction_pipeline import CustomClass


def test_custom_class_returns_dataframe():
    obj = CustomClass(
        age=39, workclass=7, education_num=13, marital_status=1,
        occupation=4, relationship=1, race=4, sex=1,
        capital_gain=2174, capital_loss=0, hours_per_week=40,
        native_country=39
    )
    df = obj.get_data_DataFrame()
    assert isinstance(df, pd.DataFrame)


def test_custom_class_dataframe_has_correct_columns():
    obj = CustomClass(
        age=39, workclass=7, education_num=13, marital_status=1,
        occupation=4, relationship=1, race=4, sex=1,
        capital_gain=2174, capital_loss=0, hours_per_week=40,
        native_country=39
    )
    df = obj.get_data_DataFrame()
    expected_cols = [
        "age", "workclass", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country"
    ]
    assert list(df.columns) == expected_cols


def test_custom_class_dataframe_has_one_row():
    obj = CustomClass(
        age=39, workclass=7, education_num=13, marital_status=1,
        occupation=4, relationship=1, race=4, sex=1,
        capital_gain=2174, capital_loss=0, hours_per_week=40,
        native_country=39
    )
    df = obj.get_data_DataFrame()
    assert len(df) == 1


def test_custom_class_values_are_correct():
    obj = CustomClass(
        age=39, workclass=7, education_num=13, marital_status=1,
        occupation=4, relationship=1, race=4, sex=1,
        capital_gain=2174, capital_loss=0, hours_per_week=40,
        native_country=39
    )
    df = obj.get_data_DataFrame()
    assert df["age"][0] == 39
    assert df["hours_per_week"][0] == 40
