import sys

from json2args import get_parameter
from json2args.data import get_data_paths

from zonal_variograms.cli import clip_dataset

# this should be json2args one day
def error(e: Exception):
    with open('/out/error.log', 'a') as f:
        f.write(str(e))
        f.write('\n')


# first load the parameters
params = get_parameter()

# get the data paths
try:
    data_paths = get_data_paths()
    raster_path = data_paths['raster']
    segments_path = data_paths['segments']
except Exception as e:
    error(e)
    print(str(e))
    sys.exit(1)

# build the command
command = []
if 'overlap' in params:
    command.append('--overlap')
if 'crs' in params:
    command.extend(['--crs', params['crs']])
if 'oid_column' in params:
    command.extend(['--oid', params['oid_column']])
if 'use_oid' in params:
    for oid in params['use_oid']:
        command.extend(['--use-oid', oid])

# finally add the raster and segments
command.extend([raster_path, segments_path])

# run the click command directly
clip_dataset(command, standalone_mode=False)