import sys
import os

from json2args import get_parameter
from json2args.data import get_data_paths

from zonal_variograms import main
from zonal_variograms.io import load_dataset, load_segments, save_to_disk

# this should be json2args one day
def error(e: Exception):
    with open('/out/error.log', 'a') as f:
        f.write(str(e))
        f.write('\n')

toolname = os.environ.get('RUN_TOOL', 'clip')

# first load the parameters
params = get_parameter()

if toolname.lower() == 'clip':
    # get the data paths
    try:
        data_paths = get_data_paths()
        raster_path = data_paths['raster']
        segments_path = data_paths['segments']
    except Exception as e:
        error(e)
        print(str(e))
        sys.exit(1)

    # load the raster
    raster = load_dataset(raster_path, crs=params.get('crs', None), raster_backend='xarray')

    # go for each row
    for layername, segment in load_segments(segments_path, crs=raster.rio.crs):
        print(f"Clipping layer {layername}...")

        clips = main.clip_features_from_dataset(raster, segment, use_oids=params.get('use_oid'), n_jobs=-1)

        # save the clips
        print(f"Clipped {len(clips)}, saving")

        save_to_disk('/out/', layername=layername, clips=clips, nested=True)

    print('\nAll done. Bye.')

else:
    err = Exception(f'Unknown tool: {toolname}')
    error(err)
    print(str(err))
    sys.exit(1)
