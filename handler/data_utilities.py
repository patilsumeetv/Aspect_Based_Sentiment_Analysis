import os
import pandas as pd

def data_reader(path):
    """
    This function is used to read the json files from the dataset.

    Parameters
    ----------
    path: str, required
        The location of review json files on the system

    Returns
    -------
    reviewText: list
        The list of the reviews present in dataset
    """

    final_df = None
    json_files = [file for file in os.listdir(path) if file.endswith('.json')]

    for file_name in json_files:
        temp_df = pd.read_json(os.path.join(path, file_name), lines=True)
        final_df = pd.concat([final_df, temp_df])

    reviewText = final_df["reviewText"].values
    return reviewText
