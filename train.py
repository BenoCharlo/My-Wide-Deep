import os
import tensorflow as tf
from core.model import Model
from core.dataset import Dataset


class WideDeepTrain(core.dataset):
    def load_data(self, dataset_dir, file_name):

        dataset_path = os.path.join(dataset_dir, file_name)
        categorical_path = os.path.join(dataset_dir, "categorical.txt")

        dataset = pd.read_csv(dataset_path)
        categorical_variables = open(categorical_path, "r").readlines()
        categorical_variables = [int(value) for value in categorical_variables]

        return dataset, categorical_variables


def train(Model):
    """Train the model"""
    dataset_train = core.dataset.Dataset()
    dataset_val = core.dataset.Dataset()

    model = Model().build_model()
