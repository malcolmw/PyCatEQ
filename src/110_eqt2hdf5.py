# USAGE: python 110_csv_to_hdf5.py INPUT_FILE_0 [INPUT_FILE_1 [INPUT_FILE_2 [...]]] OUTPUT_FILE

import argparse
import pandas as pd
import pathlib
import sys
import tqdm

def parse_clargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, nargs='+', help='Input directory.')
    parser.add_argument('output_file', type=str, help='Output file.')

    return (parser.parse_args())

def main():
    clargs = parse_clargs()
    input_files = sum(
        [
            sorted(pathlib.Path(input_dir).glob('**/*.csv'))
            for input_dir in clargs.input_dir
        ],
        []
    )

    picks = list()
    for input_file in tqdm.tqdm(sorted(input_files)):
        picks.append(pd.read_csv(input_file))

    picks = pd.concat(picks, ignore_index=True)

    for time_field in ('start_time', 'end_time', 'peak_time'):
        picks[time_field] = pd.to_datetime(picks[time_field])


    picks['network'] = picks['trace_id'].str.split('.', expand=True)[0]
    picks['station'] = picks['trace_id'].str.split('.', expand=True)[1]
    picks = picks.drop(columns=['trace_id'])
    picks = picks.rename(columns=dict(
        start_time='event_start_time',
        end_time='event_end_time',
        peak_time='arrival_time',
        peak_value='probability'
    ))
    picks = picks.sort_values(['network', 'station', 'event_start_time'])
    picks = picks.set_index(['network', 'station'])

    picks.to_hdf(clargs.output_file, key='picks')

if __name__ == '__main__':
    main()
