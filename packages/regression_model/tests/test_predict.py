import math
import os
import pathlib
from pathlib import Path
current_directory = os.path.dirname(os.path.abspath("__file__"))
parent_directory = Path(current_directory).parent

from parent_directory.regression_model.predict import make_prediction
from parent_directory.regression_model.processing.data_management import load_dataset


def test_make_single_prediction():
    # Given
    test_data = load_dataset(file_name='test.csv')
    single_test_json = test_data[0:1]

    # When
    subject = make_prediction(input_data=single_test_json)

    # Then
    assert subject is not None
    assert isinstance(subject.get('predictions')[0], float)
    assert math.ceil(subject.get('predictions')[0]) == 112476


def test_make_multiple_predictions():
    # Given
    test_data = load_dataset(file_name='test.csv')
    original_data_length = len(test_data)
    multiple_test_json = test_data

    # When
    subject = make_prediction(input_data=multiple_test_json)

    # Then
    assert subject is not None
    assert len(subject.get('predictions')) == 1451

    # We expect some rows to be filtered out
    assert len(subject.get('predictions')) != original_data_length
