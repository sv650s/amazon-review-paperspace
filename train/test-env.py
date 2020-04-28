"""
Test training python file to be used on paperspace

This program checks the following:
* data file we uploaded into /storage/amazon_reviews is available
* able to write to /artifact directory on the image

"""
import os
import sys
import pandas as pd
import numpy as np
import argparse


# do this so we can load custom util
sys.path.append("../")

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--in", dest="input",
                    help="location of input storage")
parser.add_argument("-o", "--out",dest="output",
                    help="output location. Will create models and reports directory under this"
                    )

input_dir = parser.parse_args().input
output_dir = parser.parse_args().output

data_dir = f'{input_dir}/amazon_reviews'
embeddings_dir = f'{input_dir}/embeddings'
reports_dir = f'{output_dir}/reports'
models_dir = f'{output_dir}/models'
data_file = f"{data_dir}/amazon_reviews_us_Wireless_v1_00-test-with_stop_nonlemmatized-preprocessed.csv"
out_file = f"{reports_dir}/test-output.csv"

if os.path.exists(input_dir):
    dir_list = os.listdir(input_dir)
    print(f"List {input_dir}: {dir_list}")

    dir_list = os.listdir(data_dir)
    print(f"List {data_dir}: {dir_list}")

    dir_list = os.listdir(embeddings_dir)
    print(f"List {embeddings_dir}: {dir_list}")

    print("Testing read from storage")
    if os.path.exists(data_file):
        print(f"Found {data_file}")
        df = pd.read_csv(data_file)
        print(f'df shape:\n{np.shape(df)}')
    else:
        print(f"File doesn't exist: {data_file}")
else:
    print(f'Directory missing: {input_dir}')




if os.path.exists(output_dir):
    print (f"List {output_dir}")
    os.listdir(output_dir)

    print("Testing output to artifacts")

    if not os.path.exists(reports_dir):
        print(f"{reports_dir} does not exist. Creating...")
        os.mkdir(reports_dir)
        print(f'Finished creating {reports_dir}')
    if not os.path.exists(models_dir):
        print(f"{models_dir} does not exist. Creating...")
        os.mkdir(models_dir)
        print(f'Finished creating {models_dir}')

    df_out = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df_out.to_csv(out_file, index=False)
else:
    print(f'Directory missing: {output_dir}')





