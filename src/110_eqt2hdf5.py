# USAGE: python 11_csv_to_hdf5.py INPUT_FILE_0 [INPUT_FILE_1 [INPUT_FILE_2 [...]]] OUTPUT_FILE

import argparse
import pandas as pd
import pathlib
import sys
import tqdm

def parse_clargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Input directory.")
    parser.add_argument("output_file", type=str, help="Output file.")

    return (parser.parse_args())

def main():
    clargs = parse_clargs()
    input_files = pathlib.Path(clargs.input_dir).glob("**/*.csv")

    picks = list()
    for input_file in tqdm.tqdm(sorted(input_files)):
        picks.append(pd.read_csv(input_file))

    picks = pd.concat(picks, ignore_index=True)

    for time_field in ("event_start_time", "event_end_time", "p_arrival_time", "s_arrival_time"):
        picks[time_field] = pd.to_datetime(picks[time_field])

    picks["station"] = picks["station"].str.rstrip()
    picks = picks.sort_values(["network", "station", "event_start_time"])
    picks = picks.set_index(["network", "station"])

    picks.to_hdf(clargs.output_file, key="picks")

if __name__ == "__main__":
    main()
