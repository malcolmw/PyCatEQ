import argparse
import configparser
import os
import pandas as pd
import pathlib
import subprocess
import sys
import tqdm

def parse_clargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("start_date", type=str, help="Start date.")
    parser.add_argument("end_date",   type=str, help="End date.")
    parser.add_argument(
        "-c",
        '--config_file',
        default='200_REAL.cfg',
        type=str,
        help="Configuration file."
    )

    return (parser.parse_args())

def parse_config(config_file):
    opts = dict()
    config = configparser.ConfigParser()
    config.read(config_file)
    opts['general'] = dict(
        pick_dir=pathlib.Path(config.get('general', 'pick_dir')).resolve(),
        stations=pathlib.Path(config.get('general', 'stations')).resolve(),
        real=pathlib.Path(config.get('general', 'real')).resolve(),
        tt_db=pathlib.Path(config.get('general', 'tt_db')).resolve(),
        output_dir=pathlib.Path(config.get('general', 'output_dir')).resolve()
    )
    opts['G'] = dict(
        trx=config.getfloat('grid', 'trx', fallback=None),
        trh=config.getfloat('grid', 'trh', fallback=None),
        tdx=config.getfloat('grid', 'tdx', fallback=None),
        tdh=config.getfloat('grid', 'tdh', fallback=None),
    )
    if not any(opts['G'][key] == None for key in opts['G']):
        opts['G'] = '/'.join((
            str(opts['G'][key]) for key in (
                'trx', 'trh', 'tdx', 'tdh'
            )
        ))
    else:
        opts['G'] = None
    opts['R'] = dict(
        rx=config.getfloat('search_region', 'rx'),
        rh=config.getfloat('search_region', 'rh'),
        tdx=config.getfloat('search_region', 'tdx'),
        tdh=config.getfloat('search_region', 'tdh'),
        tint=config.getfloat('search_region', 'tint'),
        gap=config.getfloat('search_region', 'gap', fallback=None),
        GCarc0=config.getfloat('search_region', 'GCarc0', fallback=None),
        latref0=config.getfloat('search_region', 'latref0', fallback=None),
        lonref0=config.getfloat('search_region', 'lonref0', fallback=None),
    )
    opt_R = '/'.join((
        str(opts['R'][key]) for key in (
            'rx', 'rh', 'tdx', 'tdh', 'tint'
        )
    ))
    for key in ('gap', 'GCarc0', 'latref0', 'lonref0'):
        opt = opts['R'][key]
        if opt is not None:
            opt_R = '/'.join((opt_R, str(opt)))
        else:
            break
    opts['R'] = opt_R

    opts['S'] = dict(
        np0=config.getint('thresholds', 'np0', fallback=4),
        ns0=config.getint('thresholds', 'ns0', fallback=2),
        nps0=config.getint('thresholds', 'nps0', fallback=6),
        npsboth0=config.getint('thresholds', 'npsboth0', fallback=2),
        std0=config.getfloat('thresholds', 'std0', fallback=0.5),
        dtps=config.getfloat('thresholds', 'dtps', fallback=0.25),
        nrt=config.getfloat('thresholds', 'nrt', fallback=1),
        drt=config.getfloat('thresholds', 'drt', fallback=0.25),
        nxd=config.getfloat('thresholds', 'nxd', fallback=0.2),
        rsel=config.getfloat('thresholds', 'nxd', fallback=0.4),
        ires=config.getboolean('thresholds', 'ires', fallback=False),
    )

    opt_S = '/'.join((
        str(opts['S'][key]) for key in (
            'np0', 'ns0', 'nps0', 'npsboth0', 'std0', 'dtps', 'nrt'
        )
    ))
    for key in ('drt', 'nxd', 'rsel', 'ires'):
        opt = opts['S'][key]
        if opt is not None and opt is not False:
            opt_S = '/'.join((opt_S, str(opt)))
        else:
            break
    opts['S'] = opt_S

    opts['V'] = dict(
        vp0=config.getfloat('velocity', 'vp0', fallback=6.2),
        vs0=config.getfloat('velocity', 'vs0', fallback=3.3),
        s_vp0=config.getfloat('velocity', 's_vp0', fallback=6.2),
        s_vs0=config.getfloat('velocity', 's_vs0', fallback=3.3),
        ielev=config.getfloat('velocity', 'ielev', fallback=0),
    )
    opt_V = '/'.join((
        str(opts['V'][key]) for key in (
            'vp0', 'vs0', 's_vp0', 's_vs0', 'ielev'
        )
    ))
    opts['V'] = opt_V

    return opts

def main():
    clargs = parse_clargs()
    opts = parse_config(clargs.config_file)
    lat0 = get_median_latitude(opts['general']['stations'])
    output_dir = opts['general']['output_dir']
    for date in pd.date_range(
        start=clargs.start_date,
        end=clargs.end_date,
        freq='D'
    ):
        path_in = opts['general']['pick_dir']
        path_in = path_in.joinpath(str(date.year), f'{date.dayofyear:03d}')
        path_out = output_dir.joinpath(str(date.year), f'{date.dayofyear:03d}')
        path_out.mkdir(parents=True, exist_ok=True)
        os.chdir(path_out)
        opts['D'] = f'{date.year}/{date.month:02d}/{date.day:02d}/{lat0}'
        cmd = filter(lambda x: x is not None, [
                'stdbuf',
                '-oL',
                opts['general']['real'],
                '-D'+opts['D'],
                '-R'+opts['R'],
                ('-G'+opts['G']) if opts['G'] is not None else None,
                '-S'+opts['S'],
                '-V'+opts['V'],
                opts['general']['stations'],
                str(path_in),
                opts['general']['tt_db']
            ]
         )
        subprocess.run(cmd)

def get_median_latitude(path):
    stations = pd.read_csv(
        path,
        names=[
            'longitude',
            'latitude',
            'network',
            'station',
            'channel',
            'elevation'
        ],
        sep=' '
    )
    stations = stations.drop_duplicates(subset=['network', 'station'])
    return stations['latitude'].median()

if __name__ == "__main__":
    main()
