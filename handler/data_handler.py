import os
import pandas as pd

def data_reader(path):
    final_df = None
    json_files = [file for file in os.listdir(path) if file.endswith('.json')]

    for file_name in json_files:
        temp_df = pd.read_json(os.path.join(path, file_name), lines=True)
        final_df = pd.concat([final_df, temp_df])

    reviewText = final_df["reviewText"].values
    return reviewText
