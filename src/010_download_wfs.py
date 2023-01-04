import argparse
import configparser
import numpy as np
import obspy
import obspy.clients
import obspy.clients.fdsn
import os
import pathlib
import sys

import my_logging
logger = my_logging.get_logger(__name__)

def parse_clargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Configuration file.")
    parser.add_argument("output_dir", type=str, help="Output directory.")
    parser.add_argument("starttime", type=str, help="Start time.")
    parser.add_argument("endtime", type=str, help="End time.")
    parser.add_argument(
        "-l",
        "--log_file",
        type=str,
        default="010_download_wfs.log",
        help="Log file."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Be verbose."
    )

    return (parser.parse_args())

def parse_config(config_file):
    converters = dict(
        time=lambda t: obspy.UTCDateTime(t)
    )
    parser = configparser.ConfigParser(converters=converters)
    parser.read(config_file)

    config = dict()

    section = parser["default"]
    config["default"] = dict(
        file_length=section.getfloat("file_length", fallback=86400),
        minimum_segment_length=section.getfloat("minimum_segment_length", fallback=0),
        base_url=section.get("base_url", fallback="IRIS")
    )

    section = parser["get_stations"]
    config["get_stations"] = dict(
        starttime=section.gettime("starttime", fallback=None),
        endtime=section.gettime("endtime", fallback=None),
        startbefore=section.gettime("startbefore", fallback=None),
        startafter=section.gettime("startafter", fallback=None),
        endbefore=section.gettime("endbefore", fallback=None),
        endafter=section.gettime("endafter", fallback=None),
        network=section.get("network", fallback=None),
        station=section.get("station", fallback=None),
        location=section.get("location", fallback=None),
        channel=section.get("channel", fallback=None),
        minlatitude=section.getfloat("minlatitude", fallback=None),
        maxlatitude=section.getfloat("maxlatitude", fallback=None),
        minlongitude=section.getfloat("minlongitude", fallback=None),
        maxlongitude=section.getfloat("maxlongitude", fallback=None),
        latitude=section.getfloat("latitude", fallback=None),
        longitude=section.getfloat("longitude", fallback=None),
        minradius=section.getfloat("minradius", fallback=None),
        maxradius=section.getfloat("maxradius", fallback=None),
        level=section.get("level", fallback=None),
        includerestricted=section.getboolean("includerestricted", fallback=None),
        includeavailability=section.getboolean("includeavailability", fallback=None),
        updatedafter=section.gettime("updatedafter", fallback=None),
        matchtimeseries=section.getboolean("matchtimeseries", fallback=None),
        filename=section.get("filename", fallback=None),
        format=section.get("format", fallback=None)
    )

    return (config)

def write_wf(stream, output_dir, minimum_segment_length):
    stream.split()
    for trace in stream:
        stats = trace.stats
        if stats.endtime - stats.starttime < minimum_segment_length:
            continue
        trace_network = stats.network
        trace_station = stats.station
        trace_location = stats.location
        trace_channel = stats.channel
        trace_year = stats.starttime.year
        trace_jday = stats.starttime.julday
        trace_starttime = stats.starttime.strftime("%Y%m%dT%H%M%SZ")
        trace_endtime = stats.endtime.strftime("%Y%m%dT%H%M%SZ")
        filename = f"{trace_network}.{trace_station}.{trace_location}.{trace_channel}__{trace_starttime}__{trace_endtime}.mseed"
        output_path = output_dir.joinpath(str(trace_year), f"{trace_jday:03d}", trace_network, trace_station, filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trace.write(str(output_path))
        print("Writing " + str(output_path) + "    ", end="\r")

def main():
    clargs = parse_clargs()
    my_logging.configure_logger(
        __name__,
        clargs.log_file,
        verbose=clargs.verbose
    )
    config = parse_config(clargs.config_file)

    client = obspy.clients.fdsn.Client(base_url=config["default"]["base_url"])
    inventory = client.get_stations(**config["get_stations"])

    starttime = obspy.UTCDateTime(clargs.starttime)
    while starttime < obspy.UTCDateTime(clargs.endtime):
        endtime = starttime + config["default"]["file_length"]
        for network in inventory.networks:
            for station in network.stations:
                #if station.code[:2] in IGNORE:
                #    continue
                for channel in config["get_stations"]["channel"].split(","):
                    channels = station.select(channel=channel).channels
                    #if len(channels) < 3:
                    #    continue
                    bulk_request = [(
                        network.code,
                        station.code,
                        "*",
                        channel,
                        starttime,
                        endtime
                    )]
                    try:
                        logger.info(f"Requesting waveforms for {network.code}.{station.code}..{channel} {starttime} - {endtime}")
                        stream = client.get_waveforms_bulk(bulk_request)
                        write_wf(
                            stream,
                            pathlib.Path(clargs.output_dir),
                            config["default"]["minimum_segment_length"]
                        )
                        logger.info(f"Successfully retrieved waveforms for {network.code}.{station.code}..{channel} {starttime} - {endtime}")
                        break
                    except Exception as exc:
                        logger.debug(exc)
                        logger.warning(f"Failed to retrieve waveforms for {network.code}.{station.code}..{channel} {starttime} - {endtime}")
        starttime = endtime

if __name__ == "__main__":
    main()
