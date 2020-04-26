"""
Test training python file to be used on paperspace

This program checks the following:
* data file we uploaded into /storage/data is available
* able to write to /artifact directory on the image

"""
import os
import sys
import pandas as pd
import argparse


# do this so we can load custom util
sys.path.append("../")

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--in", dest="input",
                    help="location of input storage")
parser.add_argument("-o", "--out",dest="output",
                    help="output location. Will create models and reports directory under this"
                    )

input = parser.parse_args().input
output = parser.parse_args().output

data_dir = f'{input}/data'
report_dir = f'{output}/reports'
data_file = f"{data_dir}/amazon_reviews_us_Wireless_v1_00-test-with_stop_nonlemmatized-preprocessed.csv"
out_file = f"{report_dir}/test-output.csv"


print("Testing read from storage")
if os.path.exists(data_file):
    df = pd.read_cvs(data_file)
else:
    print(f"File doesn't exist: {data_file}")


print("Testing output to artifacts")

if not os.path.exists(report_dir):
    print(f"{report_dir} does not exist. Creating...")
    os.mkdir(report_dir)
    print(f'Finished creating {report_dir}')

df_out = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
df_out.to_csv(out_file, index=False)



