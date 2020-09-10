import itertools
import numpy as np
import pandas as pd

from core import utils


class Dataset(object):
    """Implement Dataset here"""

    def __init__(self, dataset, categorical_indices):
        self.dataset = dataset
        self.categorical_indices = categorical_indices

        self.non_categorical_indices = [
            index
            for index in range(len(dataset.columns))
            if index not in self.categorical_indices
        ]

    def get_variables_names(self):
        categorical_variables = self.dataset.columns[self.categorical_indices]
        non_categorical_variables = self.dataset.columns[self.non_categorical_indices]

        return categorical_variables, non_categorical_variables

    def split_dataset(self):
        categorical_dataset = dataset.loc[self.categorical_indices]
        non_categorical_dataset = dataset.loc[self.non_categorical_indices]

        return categorical_dataset, non_categorical_dataset

    def cross_categorical_dataset(self):
        """Build the cross data of all categorical variables"""
        names, _ = self.get_variables_names()
        names_combinations = list(itertools.combinations(names, 2))

        output_data = []
        for combination in names_combinations:
            variable1 = list(dict.fromkeys(self.dataset[combination[0]]).keys())
            variable2 = list(dict.fromkeys(self.dataset[combination[1]]).keys())
            output_data.append(utils.cross_two_variables(variable1, variable2))

        output_data = pd.concat(output_data, axis=1)
        return output_data