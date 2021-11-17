import argparse
import numpy as np
import pandas as pd
import pathlib
import tqdm

CC_THRESH = 0.75 # Minimum cross-correlation value to retain.
DTT_MAX   = 5    # Maximum differential travel-time to retain.
NCC_MIN   = 4    # Minimum number of cross-correlation observations
                 # per event pair.


def parse_argc():
    """
    Parse and return command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "catalog",
        type=str,
        help="Input catalog."
    )
    parser.add_argument(
        "input_root",
        type=str,
        help="Input data directory."
    )
    parser.add_argument(
        "output_root",
        type=str,
        help="Output data directory."
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="HDF5",
        help="Input database format."
    )
    
    return (parser.parse_args())


def read_catalog(argc):
    
    if argc.format.upper() == "HDF5":

        return (_read_catalog_hdf5(argc.catalog))

    elif argc.format.upper() == "ANTELOPE":

        return (_read_catalog_antelope(argc.catalog))

    else:

        raise (NotImplementedError)
        
        
def read_cc(argc, events):
    

    dataf = pd.DataFrame()

    desc = "Loading differential travel times"
    for path in tqdm.tqdm(sorted(pathlib.Path(argc.input_root).iterdir()), desc=desc):
        try:
            df = pd.read_hdf(path, key="differentials")
        except Exception as err:
            print(err)
            continue
        df = df[
            (df["ccmax"].abs() > CC_THRESH)
        ]
        df = df.groupby(["event_id_A", "event_id_B", "network", "station", "phase"])
        df = df.mean()
        df = df.reset_index()
        dataf = dataf.append(df, ignore_index=True)

    try:
        dataf["origin_time_A"] = events.loc[dataf["event_id_A"].values, "time"].values
        dataf["origin_time_B"] = events.loc[dataf["event_id_B"].values, "time"].values
    except KeyError:
        print(
            "Event IDs are missing from the input database; make sure it is"
            "the same database used to extract waveforms."
        )
        raise
        
    dataf["dtt"] = dataf["dt"] - (dataf["origin_time_B"] - dataf["origin_time_A"])
    dataf = dataf[
        (dataf["ccmax"].abs() > CC_THRESH)
    ]
    dataf = dataf.groupby(["event_id_A", "event_id_B", "network", "station", "phase"])
    dataf = dataf.mean()

    dataf = dataf[dataf["dtt"].abs() < DTT_MAX]

    dataf.reset_index(inplace=True)
    dataf.set_index(["event_id_A", "event_id_B"], inplace=True)
    dataf.sort_index(inplace=True)
    
    return (dataf)


def subset_observations(dataf, events, ncc_min=NCC_MIN):
    
    group = dataf.groupby(["event_id_A", "event_id_B"])
    group = group.size()
    group = group[group >= ncc_min]
    dataf = dataf.loc[group.index]
    
    event_ids = dataf.reset_index()[["event_id_A", "event_id_B"]].values.flatten()
    event_ids = np.sort(np.unique(event_ids))
    events = events.loc[event_ids]
    
    return (dataf, events)


def write_dtcc(dataf, path):
    path = pathlib.Path(path).joinpath("dt.cc")
    path.parent.mkdir(parents=True, exist_ok=True)
    dataf["station"] = dataf["station"].map("{:>5s}".format)
    dataf["dtt"]     = dataf["dtt"].map("{:>6.3f}".format)
    dataf["ccmax"]   = dataf["ccmax"].abs().map("{:>5.3f}".format)
    desc = "Writing dt.cc file."
    with open(path, "w") as outfile:
        for event_id_A, event_id_B in tqdm.tqdm(dataf.index.unique(), desc=desc):
            chunk = f"# {event_id_A:>6d} {event_id_B:>6d}    0.0\n"
            df = dataf.loc[(event_id_A, event_id_B)]
            chunk += "\n".join(
                "  " + df["station"]
                + "   " + df["dtt"]
                + "   " + df["ccmax"]
                + "   " + df["phase"]
            )
            outfile.write(chunk + "\n")


def write_events(events, path):
    path = pathlib.Path(path).joinpath("events.gc")
    path.parent.mkdir(parents=True, exist_ok=True)
    desc = "Writing events.gc file"
    with open(path, mode="w") as outfile:
        for event_id, event in tqdm.tqdm(events.iterrows(), total=len(events), desc=desc):
            latitude, longitude, depth, time = event
            time  = pd.to_datetime(time*1e9)
            line  = f"{event_id:>6d} {time.year:>4d} {time.month:>2d} {time.day:>2d} {time.hour:>2d} "
            line += f"{time.minute:>2d} {time.second:>2d}.{time.microsecond:06d} "
            line += f"{latitude:>9.6f} {longitude:>10.6f} {depth:>6.3f} -1 -1 -1 -1"
            outfile.write(line + "\n")


def _read_catalog_antelope(path):

    TABLES = dict(
        event=[
            "evid", "evname", "prefor", "auth", "commid", "lddate"
        ],
        origin=[
            "lat", "lon", "depth", "time", "orid", "evid", "jdate", "nass", "ndef",
            "ndp", "grn", "srn", "etype", "UNKNOWN", "depdp", "dtype", "mb", "mbid", "ms",
            "msid", "ml", "mlid", "algorithm", "auth", "commid", "lddate"
        ]
    )

    db = dict()

    for table in TABLES:
        db[table] = pd.read_csv(
            f"{path}.{table}",
            header=None,
            delim_whitespace=True,
            names=TABLES[table]
        )

    events = db["event"].merge(
        db["origin"][["lat", "lon", "depth", "time", "orid"]],
        left_on="prefor",
        right_on="orid"
    )

    events = events[["lat", "lon", "depth", "time", "evid"]]
    events = events.rename(
        columns=dict(
            lat="latitude",
            lon="longitude",
            evid="event_id"
        )
    )

    events = events.sort_values("event_id")
    events = events.reset_index(drop=True)

    return (events)


def _read_catalog_hdf5(path):

    events = pd.read_hdf(path, key="events")
    events = events.reset_index(drop=True)

    return (events)


def main():
    argc           = parse_argc()
    events         = read_catalog(argc)
    events         = events.sort_values("event_id")
    events         = events.set_index("event_id")
    dataf          = read_cc(argc, events)
    dataf, events  = subset_observations(dataf, events)
    write_dtcc(dataf, argc.output_root)
    write_events(events, argc.output_root)


if __name__ == "__main__":
    main()
