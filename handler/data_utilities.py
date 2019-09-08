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
    final_df: pandas dataframe
        The dataframe of all readed json files from given location
    """

    final_df = None
    json_files = [file for file in os.listdir(path) if file.endswith('.json')]

    for file_name in json_files:
        temp_df = pd.read_json(os.path.join(path, file_name), lines=True)
        final_df = pd.concat([final_df, temp_df])

    return final_df
