import argparse
import h5py
import numpy as xp
import multiprocessing as mp
import pandas as pd
import pathlib
import tqdm as tqdm
import warnings

warnings.filterwarnings("ignore")

TEMPLATE_TLEAD_P = 0.25
TEMPLATE_TLAG_P  = 0.75
TEMPLATE_TLEAD_S = 0.25
TEMPLATE_TLAG_S  = 1.25
NPAIRS           = 200
CC_THRESHOLD     = 0.7
CC_ABSOLUTE      = True


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
        "network",
        type=str,
        help="Network code."
    )
    parser.add_argument(
        "station",
        type=str,
        help="Station code."
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
        "-d",
        "--maximum_distance",
        type=float,
        default=0.1,
        help="Maximum epicentral distance to correlate."
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="HDF5",
        help="Input database format."
    )
    parser.add_argument(
        "-n",
        "--nproc",
        type=int,
        help="Number of simultaneous processes. Defaults to os.cpu_count()"
    )
    parser.add_argument(
        "-p",
        "--phases",
        type=str,
        default="P,S",
        help="Phases to process."
    )

    argc = parser.parse_args()

    argc.catalog = pathlib.Path(argc.catalog)
    argc.input_root = pathlib.Path(argc.input_root)
    argc.output_root = pathlib.Path(argc.output_root)

    argc.phases = argc.phases.split(",")

    return (argc)


def cross_correlate(template, test):
    nsamp = len(template)
    shape = (test.size - nsamp + 1, nsamp)
    rolling = xp.lib.stride_tricks.as_strided(
        test,
        shape=shape,
        strides=(test.itemsize, test.itemsize)
    )
    norm = xp.sqrt(xp.sum(xp.square(template))) \
        * xp.sqrt(xp.sum(xp.square(rolling), axis=1))
    cc = xp.correlate(test, template, mode="valid") / norm

    return (cc)


def correlate_event_pair(args):
    template_wfs, test_wfs = args
    idx_max, cc_max = list(), list()

    for i in range(3):
        cc = cross_correlate(
            template_wfs[:, i].copy(), # Copy is necessary to avoid using
            test_wfs[:, i].copy()      # corrupt memory provided by
        )                              # xp.lib.stride_tricks.as_strided.

        if CC_ABSOLUTE is True:
            idx = xp.argmax(xp.abs(cc))
        else:
            idx = xp.argmax(cc)
        idx_max.append(idx)
        cc_max.append(cc[idx])

    return(xp.array(idx_max), xp.array(cc_max))


def __correlate_event_pair(args): # Revised from above to implement subsample precision
    NSAMP_INTERP = 2

    template_wfs, test_wfs = args
    idx_max, cc_max = list(), list()

    for i in range(3):
        cc = cross_correlate(
            template_wfs[:, i].copy(), # Copy is necessary to avoid using
            test_wfs[:, i].copy()      # corrupt memory provided by
        )                              # xp.lib.stride_tricks.as_strided.

        if CC_ABSOLUTE is True:
            idx = xp.argmax(xp.abs(cc))
        else:
            idx = xp.argmax(cc)
        cc_max.append(cc[idx])

        if idx > NSAMP_INTERP and idx < len(cc)-NSAMP_INTERP:
            segment = xp.sign(cc[idx]) * cc[idx-NSAMP_INTERP: idx+NSAMP_INTERP+1]
            x = xp.arange(idx-NSAMP_INTERP, idx+NSAMP_INTERP+1)
            a, b, c = xp.polyfit(x, segment, 2)
            idx = -b / (2 * a)

        idx_max.append(idx)

    return(xp.array(idx_max), xp.array(cc_max))


def read_catalog(argc):

    if argc.format.upper() == "HDF5":

        return (_read_catalog_hdf5(argc.catalog))

    elif argc.format.upper() == "ANTELOPE":

        return (_read_catalog_antelope(argc.catalog))

    else:

        raise (NotImplementedError)


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

    events = events.reset_index(drop=True)

    return (events)


def _read_catalog_hdf5(path):

    events = pd.read_hdf(path, key="events")
    events = events.reset_index(drop=True)

    return (events)


def initialize_output(path, nevents, npair):
    f5 = h5py.File(path, mode="w")
    f5.create_dataset("event_ids", shape=(nevents * npair, 2), dtype=xp.int32)
    f5.create_dataset("cc_max", shape=(nevents * npair,), dtype=xp.float16)
    f5.create_dataset("dt", shape=(nevents * npair,), dtype=xp.float64)

    return (f5)

def correlate_events(phase, events, npairs, f5in, nproc, maximum_distance):
    INPUT_NSAMP_LEAD     = f5in[phase].attrs["nsamp_lead"]
    INPUT_NSAMP_LAG      = f5in[phase].attrs["nsamp_lag"]
    SAMPLING_RATE        = f5in[phase].attrs["sampling_rate"]
    TEMPLATE_TLEAD       = globals()[f"TEMPLATE_TLEAD_{phase}"]
    TEMPLATE_TLAG        = globals()[f"TEMPLATE_TLAG_{phase}"]
    TEMPLATE_ISAMP_START = INPUT_NSAMP_LEAD - int(TEMPLATE_TLEAD * SAMPLING_RATE)
    TEMPLATE_ISAMP_END   = TEMPLATE_ISAMP_START + int(TEMPLATE_TLAG * SAMPLING_RATE)
    CHANNEL_MAP          = {0: "Z", 1: "H1", 2: "H2"}

    events["dist"] = xp.inf

    results, output = list(), list()
    pbar = tqdm.tqdm(total=len(events))

    with mp.Pool(nproc) as pool:
        while len(events) > 1:
            pbar.update(1)
            template_event = events.iloc[0]
            events = events.iloc[1:] # Pop the template event.

            events["dist"] = xp.sqrt(
                  xp.square(template_event["latitude"] - events["latitude"])
                + xp.square(template_event["longitude"] - events["longitude"])
            )
            events = events.sort_values("dist")

            template_event_id  = int(template_event["event_id"])
            template_event_idx = int(template_event["event_idx"])
            template_wfs       = f5in[phase][
                template_event_idx,
                TEMPLATE_ISAMP_START: TEMPLATE_ISAMP_END,
                :
            ]
            template_raw_starttimes = f5in[f"starttime_{phase}"][template_event_idx]

            test_events = events.iloc[:npairs].sort_values("event_idx")
            test_events = test_events[test_events["dist"] <= maximum_distance]

            if len(test_events) == 0:
                continue

            test_event_ids = test_events["event_id"].values
            test_event_idxs = test_events["event_idx"].values


            args = ( (template_wfs, f5in[phase][test_event_idx, :, :])
                for test_event_idx in test_event_idxs
            )

            results = pool.map(correlate_event_pair, args)

            argmax = xp.array([result[0] for result in results])
            ccmax  = xp.array([result[1] for result in results])

            test_raw_starttimes = f5in[f"starttime_{phase}"][
                test_event_idxs
            ]

            _ccmax                   = xp.abs(ccmax) if CC_ABSOLUTE is True else ccmax
            mask                     = xp.nonzero(_ccmax > CC_THRESHOLD)
            event_mask, channel_mask = mask

            template_starttimes = template_raw_starttimes[channel_mask] \
                - TEMPLATE_TLEAD \
                + INPUT_NSAMP_LEAD / SAMPLING_RATE
            test_starttimes     = test_raw_starttimes[mask] \
                + argmax[mask] / SAMPLING_RATE
            dts                 = test_starttimes - template_starttimes

            output.append(
                {
                    "event_id_A": xp.repeat(template_event_id, len(event_mask)),
                    "event_id_B": test_event_ids[event_mask],
                    "ccmax": ccmax[mask],
                    "dt": dts,
                    "component": [CHANNEL_MAP[channel_idx] for channel_idx in channel_mask]

                }
            )

    pbar.update(1)
    pbar.close()

    output = pd.concat((pd.DataFrame(o) for o in output), ignore_index=True)

    return (output)

def main():
    argc = parse_argc()
    input_path = argc.input_root.joinpath(
        ".".join((argc.network, argc.station, "h5"))
    )
    EVENTS = read_catalog(argc)
    print(EVENTS)
    exit()
    argc.output_root.mkdir(exist_ok=True, parents=True)

    with h5py.File(str(input_path), mode="r") as f5in:
        for phase in argc.phases:
            mask = f5in[f"mask_{phase}"][:]
            EVENTS["event_idx"] = EVENTS.index.values
            events = EVENTS[mask]

            assert (xp.all(EVENTS["event_id"] == f5in["event_id"][:])), \
                "Incompatible catalog and waveforms."

            nproc = mp.cpu_count() if argc.nproc is None else argc.nproc
            print(f"Processing {phase}-wave data with {nproc} processes...")
            output = correlate_events(
                phase,
                events,
                NPAIRS,
                f5in,
                nproc=nproc,
                maximum_distance=argc.maximum_distance
            )

            path = argc.output_root.joinpath(
                ".".join(( argc.network, argc.station, phase, "h5"))
            )
            output["network"] = argc.network
            output["station"] = argc.station
            output["phase"] = phase
            print(f"Save {phase}-wave differential travel times to {path}.")
            try:
                output.to_hdf(path, key="differentials", format="table")
            except:
                path.unlink()
                path = path.parent.joinpath(path.name.replace("h5", "hdf5"))
                with h5py.File(path, mode="w") as f5:
                    for field in output.columns:
                        f5.create_dataset(field, data=output[field].values)


if __name__ == "__main__":
    main()
