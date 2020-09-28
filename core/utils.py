import pandas as pd

#################################
# Dataset utilities
#################################


def load_data(dataset_dir, file_name):

    dataset_path = os.path.join(dataset_dir, file_name)
    categorical_path = os.path.join(dataset_dir, "categorical.txt")
    target_path = os.path.join(dataset_dir, "target.txt")

    dataset = pd.read_csv(dataset_path)
    categorical_variables = open(categorical_path, "r").readlines()
    categorical_variables = [int(value) for value in categorical_variables]
    target_name = open(target_path, "r").readlines()[0]

    return dataset, target_name, categorical_variables


def cross_two_variables(variable1, variable2):
    """
    Create a cross dataset from categories from the 2 variables
    Variables should be in a form of list
    """

    assert len(variable1) == len(variable2)
    names = []
    interactions = []

    names = [variable1[i] + "_" + variable2[i] for i in range(len(variable1))]

    for name in names:
        interaction = []
        interaction += [
            1
            if (
                variable1[i] + "_" + variable2[i]
                == name.split("_")[0] + "_" + name.split("_")[1]
            )
            else 0
            for i in range(len(variable1))
        ]
        interactions.append(interaction)

    output_data = pd.DataFrame.from_records(interactions)
    output_data.columns = names

    return output_data
