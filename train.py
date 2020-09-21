import os
import tensorflow as tf
from core.model import Model
from core.dataset import Dataset
import core.utils as utils


def train(Model):
    """Train the model"""

    train_set, categoricals = utils.load_data(args.dataset, args.file_name)
    dataset_train = Dataset(dataset=train_set, categorical_indices=categoricals)
    (
        categorical_variables,
        non_categorical_variables,
    ) = dataset_train.get_variables_names()

    print("Creating cross products dataset")
    cross_products = dataset_train.cross_categorical_dataset()

    model = Model(categorical_variables, non_categorical_variables)
    model = model.build_model(cross_products)

    print("Training model")
    model.fit(
        [
            [train_set[categorical_variables], train_set[non_categorical_variables]],
            cross_products,
        ],
    )
