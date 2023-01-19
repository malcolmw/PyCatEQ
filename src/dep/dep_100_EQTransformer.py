# USAGE: python 100_pick.py INPUT_ROOT_DIR YEAR JDAY

import argparse
import configparser
import json
import os
import pathlib
import tempfile
import warnings
warnings.filterwarnings("ignore")

import EQTransformer.core.mseed_predictor


def parse_clargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Input data directory.")
    parser.add_argument("output_dir", type=str, help="Output data directory.")
    parser.add_argument("model", type=str, help="EQTransformer model weights.")
    parser.add_argument("-c", "--config", type=str, help="Configuration file.")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU.")

    return (parser.parse_args())


def parse_config(clargs):
    parser = configparser.ConfigParser()

    if clargs.config:
        parser.read(clargs.config)

    parser = parser["DEFAULT"]
    config = dict(
        detection_threshold=parser.getfloat("detection_threshold", fallback=0.3),
        P_threshold=parser.getfloat("P_threshold", fallback=0.1),
        S_threshold=parser.getfloat("S_threshold", fallback=0.1),
        number_of_plots=parser.getint("number_of_plots", fallback=0),
        plot_mode=parser.get("plot_mode", fallback="time"),
        overlap=parser.getfloat("overlap", fallback=0.3),
        batch_size=parser.getint("batch_size", fallback=500)

    )

    return (config)


def main():

    clargs = parse_clargs()
    config = parse_config(clargs)

    if clargs.gpu:
        print("Running on GPU(s)")
    else:
        print("Running on CPU(s)")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    input_dir = pathlib.Path(clargs.input_dir)
    output_dir = pathlib.Path(clargs.output_dir)

    for network_dir in sorted(input_dir.iterdir()):
        stations = build_stations_dict(network_dir)
        with tempfile.NamedTemporaryFile(mode="r+") as tmpfile:
            json.dump(stations, tmpfile)
            tmpfile.flush()
            output_dir = output_dir.joinpath(
                network_dir.name
            )
            EQTransformer.core.mseed_predictor(
                input_model=clargs.model,
                input_dir=str(network_dir),
                output_dir=str(output_dir),
                stations_json=tmpfile.name,
                **config
            )


def build_stations_dict(network_dir):
    print("Building station dict...")
    stations = dict()
    for station_dir in sorted(network_dir.iterdir()):
        for dfile in sorted(station_dir.iterdir()):
            network, station, _, channel = dfile.name.split("__")[0].split(".")
            if station not in stations:
                stations[station] = dict(
                    network=network,
                    channels=[channel],
                    coords=[-1, -1, -1]
                )
            elif channel not in stations[station]["channels"]:
                stations[station]["channels"].append(channel)

    return (stations)


if __name__ == "__main__":
    main()
