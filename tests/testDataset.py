import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

import core.utils as utils
import core.dataset as dataset

languages = ["EN", "FR"]
age = [13, 23]
gender = ["M", "F"]
height = [1.75, 1.69]
job = ["STUDENT", "WORKER"]

my_dataset = pd.DataFrame(
    {"languages": languages, "age": age, "gender": gender, "height": height, "job": job}
)

my_dataset = dataset.Dataset(my_dataset, [0, 2, 4])


class TestDataset(unittest.TestCase):
    def test_cross_two_variables(self):

        languages = ["EN", "FR"]
        gender = ["M", "F"]
        names = ["EN_M", "FR_F"]
        interactions = [[1, 0], [0, 1]]
        excepted_output = pd.DataFrame.from_records(interactions)
        excepted_output.columns = names

        new_dataset = utils.cross_two_variables(languages, gender)
        assert_frame_equal(excepted_output, new_dataset, check_dtype=False)

    def test_cross_categorical_dataset(self):

        languages = ["EN", "FR"]
        gender = ["M", "F"]
        job = ["STUDENT", "WORKER"]
        names = ["EN_M", "FR_F", "EN_STUDENT", "FR_WORKER", "M_STUDENT", "F_WORKER"]
        interactions = [[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]]
        excepted_output = pd.DataFrame.from_records(interactions)
        excepted_output.columns = names

        new_dataset = my_dataset.cross_categorical_dataset()
        assert_frame_equal(excepted_output, new_dataset, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
