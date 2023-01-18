import argparse
import configparser
import os
import pandas as pd
import pathlib
import subprocess

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
    config = dict()
    parser = configparser.ConfigParser()
    parser.read(config_file)
    config['general'] = dict(
        pick_dir=pathlib.Path(parser.get('general', 'pick_dir')).resolve(),
        stations=pathlib.Path(parser.get('general', 'stations')).resolve(),
        real=pathlib.Path(parser.get('general', 'real')).resolve(),
        tt_db=pathlib.Path(parser.get('general', 'tt_db')).resolve(),
        output_dir=pathlib.Path(parser.get('general', 'output_dir')).resolve()
    )
    config['G'] = dict(
        trx=parser.getfloat('grid', 'trx', fallback=None),
        trh=parser.getfloat('grid', 'trh', fallback=None),
        tdx=parser.getfloat('grid', 'tdx', fallback=None),
        tdh=parser.getfloat('grid', 'tdh', fallback=None),
    )
    if not any(config['G'][key] == None for key in config['G']):
        config['G'] = '/'.join((
            str(config['G'][key]) for key in (
                'trx', 'trh', 'tdx', 'tdh'
            )
        ))
    else:
        config['G'] = None
    config['R'] = dict(
        rx=parser.getfloat('search_region', 'rx'),
        rh=parser.getfloat('search_region', 'rh'),
        tdx=parser.getfloat('search_region', 'tdx'),
        tdh=parser.getfloat('search_region', 'tdh'),
        tint=parser.getfloat('search_region', 'tint'),
        gap=parser.getfloat('search_region', 'gap', fallback=None),
        GCarc0=parser.getfloat('search_region', 'GCarc0', fallback=None),
        latref0=parser.getfloat('search_region', 'latref0', fallback=None),
        lonref0=parser.getfloat('search_region', 'lonref0', fallback=None),
    )
    opt_R = '/'.join((
        str(config['R'][key]) for key in (
            'rx', 'rh', 'tdx', 'tdh', 'tint'
        )
    ))
    for key in ('gap', 'GCarc0', 'latref0', 'lonref0'):
        opt = config['R'][key]
        if opt is not None:
            opt_R = '/'.join((opt_R, str(opt)))
        else:
            break
    config['R'] = opt_R

    config['S'] = dict(
        np0=parser.getint('thresholds', 'np0', fallback=4),
        ns0=parser.getint('thresholds', 'ns0', fallback=2),
        nps0=parser.getint('thresholds', 'nps0', fallback=6),
        npsboth0=parser.getint('thresholds', 'npsboth0', fallback=2),
        std0=parser.getfloat('thresholds', 'std0', fallback=0.5),
        dtps=parser.getfloat('thresholds', 'dtps', fallback=0.25),
        nrt=parser.getfloat('thresholds', 'nrt', fallback=1),
        drt=parser.getfloat('thresholds', 'drt', fallback=0.25),
        nxd=parser.getfloat('thresholds', 'nxd', fallback=0.2),
        rsel=parser.getfloat('thresholds', 'nxd', fallback=0.4),
        ires=parser.getboolean('thresholds', 'ires', fallback=False),
    )

    opt_S = '/'.join((
        str(config['S'][key]) for key in (
            'np0', 'ns0', 'nps0', 'npsboth0', 'std0', 'dtps', 'nrt'
        )
    ))
    for key in ('drt', 'nxd', 'rsel', 'ires'):
        opt = config['S'][key]
        if opt is not None and opt is not False:
            opt_S = '/'.join((opt_S, str(opt)))
        else:
            break
    config['S'] = opt_S

    config['V'] = dict(
        vp0=parser.getfloat('velocity', 'vp0', fallback=6.2),
        vs0=parser.getfloat('velocity', 'vs0', fallback=3.3),
        s_vp0=parser.getfloat('velocity', 's_vp0', fallback=6.2),
        s_vs0=parser.getfloat('velocity', 's_vs0', fallback=3.3),
        ielev=parser.getfloat('velocity', 'ielev', fallback=0),
    )
    opt_V = '/'.join((
        str(config['V'][key]) for key in (
            'vp0', 'vs0', 's_vp0', 's_vs0', 'ielev'
        )
    ))
    config['V'] = opt_V

    return config

def main():
    clargs = parse_clargs()
    config = parse_config(clargs.config_file)
    lat0 = get_median_latitude(config['general']['stations'])
    output_dir = config['general']['output_dir']
    for date in pd.date_range(
        start=clargs.start_date,
        end=clargs.end_date,
        freq='D'
    ):
        path_in = config['general']['pick_dir']
        path_in = path_in.joinpath(str(date.year), f'{date.dayofyear:03d}')
        path_out = output_dir.joinpath(str(date.year), f'{date.dayofyear:03d}')
        path_out.mkdir(parents=True, exist_ok=True)
        os.chdir(path_out)
        config['D'] = f'{date.year}/{date.month:02d}/{date.day:02d}/{lat0}'
        cmd = filter(lambda x: x is not None, [
                'stdbuf',
                '-oL',
                config['general']['real'],
                '-D'+config['D'],
                '-R'+config['R'],
                ('-G'+config['G']) if config['G'] is not None else None,
                '-S'+config['S'],
                '-V'+config['V'],
                config['general']['stations'],
                str(path_in),
                config['general']['tt_db']
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
