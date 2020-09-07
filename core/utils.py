import pandas as pd

#################################
# Dataset utilities
#################################


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
