import tensorflow as tf
from tf import keras

from core import utils, config


class Model(object):

    def __init__(self, embedding_size):
        pass

    
    def build_model(self, categorical_variables, non_categorical_variables, cross_products):

        categorical_input = self.input_fn(categorical_variables)
        non_categorical_input = self.input_fn(non_categorical_variables, False)

        embeddings = self.input_embedding(categorical_input)

        flattens = self.input_flattened(embeddings)
        formatted_input = list(non_categorical_input.values()) + list(flattens.values())
        concatenate_1 = tf.keras.layers.Concatenate()(formatted_input)

        dense_1 = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(concatenate)
        dense_2 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(dense_1)
        dense_3 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(dense_2)

        flatten_cross_products = tf.keras.layers.Flatten()(cross_products)

        concatenate_2 = tf.keras.layers.Concatenate()([dense_3, flatten_cross_products])

        output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(concatenate_2)

        model = tf.keras.Model(inputs=[[categorical_variables, non_categorical_variables], cross_products], outputs=output)

        model.compile(optimizer=tf.train.AdamOptimizer(config.LEARNING_RATE),
                    loss=tf.keras.losses.binary_crossentropy,
                    metrics=[tf.keras.metrics.Accuracy])

        return model
    
    def input_fn(self, variables, is_categorical=True):
        input_variables = dict()
        
        if is_categorical:
            for variable in variables:
                var_shape = len(np.unique(variable))
                input_variables[str(variable)] = (tf.keras.layers.Input(shape=len(np.unique(variable))), var_shape)

        else:
            var_shape = len(variables)
            input_variables["non_categorical_input"] = tf.keras.layers.Input(shape=var_shape)

        return input_variables

    def input_embedding(self, variables_input):

        embeddings = dict()
        for key in variables_input.keys():
            variable, var_shape = variables_input[key][0], variables_input[key][1]]
            
            embeddings[key] = tf.keras.layers.Embedding(
                name="embedding_" + str(key), 
                input_dim=var_shape,
                output_dim = config.EMBEDDING_SIZE)(variable)

        return embeddings

    def input_flattened(self, embeddings):

        flattened = dict()
        for key in embeddings.keys():
            embedding = embeddings[key]
            flattened["flattened" + str(key.split("_")[1])] = tf.keras.layers.Flatten()(embedding)

        return flattened

###########################
# Utility functions
###########################


def categorical_embedding(input_data, variable_name, size_input_data, embedding_size):

    with tf.variable_scope(variable_name):
        my_variable = keras.layers.Input(name=variable_name, shape=[1])

        embedding = keras.layers.Embedding(
            name="embedding" + str(variable_name),
            input_dim=size_input_data,
            output_dim=embedding_size,
        )(my_variable)

    return embedding


def create_cross_products(CATEGORICAL_VARIABLES):
    