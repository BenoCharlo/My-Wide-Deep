import os
import tensorflow as tf
from core.model import Model
from core.dataset import Dataset


class WideDeepTrain(core.dataset):
    


def train(Model):
    """Train the model"""
    dataset_train = core.dataset.Dataset()
    dataset_val = core.dataset.Dataset()

    model = Model().build_model()
