'''
ingestion.py

Author: Wonseok Oh
Date: June 2023
'''
import os
import json

import pandas as pd

# Load config.json and get input and output paths
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


# Function for data ingestion
def merge_multiple_dataframe():
    '''
    merge_multiple_dataframe

    Input: None
    Output: pandas.DataFrame

    check for datasets, compile them together,
    and write to an output file
    '''
    combined_data = pd.DataFrame()
    file_names = []
    for file_name in os.listdir(input_folder_path):
        file_path = os.path.join(input_folder_path, file_name)

        if file_name.endswith('.csv'):
            data = pd.read_csv(file_path)
            combined_data = combined_data.append(data, ignore_index=True)
            file_names.append(file_name)

    combined_data.drop_duplicates(inplace=True)

    output_file_path = os.path.join(
        output_folder_path, 'finaldata.csv')
    record_full_path = os.path.join(
        output_folder_path,
        'ingestedfiles.txt')
    combined_data.to_csv(output_file_path, index=False)

    with open(record_full_path, 'w', encoding='utf-8') as f_p:
        for file_name in file_names:
            f_p.write(file_name + '\n')


# main function
if __name__ == '__main__':
    merge_multiple_dataframe()
