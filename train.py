import os
import tensorflow as tf
import core


class WideDeepTrain(core.dataset):
    def load_data(self, dataset_dir, subset):

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)