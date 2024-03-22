from simwave.kernel.frontend.dataset_writer import DatasetWriter
import pytest
import numpy as np
import os


class TestDatasetWriter:

    def test_file_is_created(self):

        data = {
            "my_dataset_1": {
                "dataset_data": np.array([1, 2, 3]),
                "dataset_attributes": {
                    "description": "small numbers",
                    "location": "collected at lab X",
                },
            },
            "my_dataset_2": {
                "dataset_data": np.array([3, 2, 1]),
                "dataset_attributes": {
                    "my_attribute_1": "small numbers",
                    "my_attribute_2": "collected at lab Y",
                },
            },
        }
        path = "tmp/pytest_test_data.h5"
        DatasetWriter.write_dataset(data, path)
        assert os.path.isfile(path)
