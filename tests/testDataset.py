import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

import core.utils as utils


class TestDataset(unittest.TestCase):
    def test_cross_two_variables(self):

        languages = ["EN", "FR"]
        gender = ["M", "F"]
        names = ["EN_M", "FR_F"]
        interactions = [[1, 0], [0, 1]]
        excepted_output = pd.DataFrame.from_records(interactions)
        excepted_output.columns = names

        new_dataset = utils.cross_two_variables(languages, gender)
        print(new_dataset.shape)
        print(excepted_output.shape)
        assert_frame_equal(excepted_output, new_dataset, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
