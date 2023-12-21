import os
from pathlib import Path

import click
import rasterio
from rasterio.errors import RasterioIOError
import geopandas as gpd
import pandas as pd
import fiona
import netCDF4
from fiona.errors import DriverError
from pyproj import CRS
import matplotlib.pyplot as plt
import numpy as np

from zonal_variograms.backend.rasterio import add_variograms_to_segmentation, add_univariate_to_segmentation, get_raster_band, clip_features


@click.command(context_settings=dict(help_option_names=['-h', '--help'], ignore_unknown_options=True, allow_extra_args=True))
@click.option('--dtype', default=None, type=str, help='Unsafe cast the raster to this dtype. If empty, the original data type of the raster will be used.')
@click.option('--ignore-crs', default=False, is_flag=True, help="Ignore the CRS of the raster and the segments. This may lead to wrong results. Can be used for non-referenced files.")
@click.option('--sample', default=None, type=int, help="Sample size to sample the zones for the variograms. If empty, all data will be used.")
@click.option('--seed', default=None, type=int, help="Seed for the random number generator if zones are resampled.")
@click.option('--model', default='spherical', help="The variogram model to use.")
@click.option('--n-lags', default=10, type=int, help="The number of lag classes for the empirical variogram.")
@click.option('--maxlag', default=None, help="The maximum search distance for the variogram")
@click.option('--use-nugget', default=False, is_flag=True, help="Use a nugget for the model")
@click.option('--quiet', default=False, is_flag=True, help="Suppress all output.")
@click.option('--add-clip', default=False, is_flag=True, help="Add the clipped zone as image to the properties. Slows everything down.")
@click.option('--use-band', default=None, type=(int, str), help="The band to use from the raster dataset. Defaults to 0.")
@click.option('--output-file', default=None, help="The output file to write the results to. If empty, the results will be written to the segments file.")
@click.option('--stats-to-file', default=False, is_flag=True, help="Write the univariate statistics to a CSV file. This is useful if you do not want to aggregate bands to a single number per feature.")
@click.option('--axis', default=None, type=int, multiple=True, help="The axis for band aggregation within a clip feature, to aggregate along that axis. Defaults to None for full aggregation. Can be passed multiple times.")
@click.option('--skip-img', default=False, is_flag=True, help="Skip the creation of the images. This is useful if you only want the result files")
@click.option('--add-json', default=False, is_flag=True, help="Output the data into a json file for each layer as well.")
@click.argument('raster')
@click.argument('segments')
@click.pass_context
def process_segmented_files(ctx, dtype, ignore_crs, sample, seed, model, n_lags, maxlag, use_nugget, quiet, add_clip, use_band, output_file, stats_to_file, axis, skip_img, add_json, raster, segments):
    """
    Calculate zonal variograms of RASTER for each segment in SEGMENTS.

    RASTER has to be a raster file with only one band containing the data. If more are found
    the first band will be used.\n
    SEGMENTS has to be a shapefile or geopackage container one or many layer of Polygons 
    identifying the zones. If more than one layer is found, the statistics will be calculated for 
    all of them.
    """
    raise NotImplementedError("The CLI has to be updated to the new API first. Sorry.")
    # open the raster
    try:
        raster: rasterio.DatasetReader = rasterio.open(raster)
    except RasterioIOError as e:
        click.echo(f"{raster} is not a valid raster file.\nDetails: {str(e)}")
        return

    # get all layers on the segments
    try:
        layernames = [name for name in fiona.listlayers(segments) if name != 'layer_styles']
    except DriverError as e:
        click.echo(str(e))
        return
    
    # build a basename for the output
    if output_file is None:
        output_file = os.path.join(os.path.dirname(segments),Path(segments).stem , f"{Path(segments).stem}_with_variograms.gpkg")
    if not output_file.endswith('.gpkg'):
        output_file = os.path.join(output_file, f"{Path(output_file).stem}_with_variograms.gpkg")
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # some stats about the raster
    if not quiet:
        click.echo(f"INPUT RASTER")
        click.echo(f"-----------")
        click.echo(f"HEIGHT:   {raster.height}")
        click.echo(f"WIDTH:    {raster.width}")
        click.echo(f"CRS:      {raster.crs}")
        click.echo(f"BOUNDS:   {raster.bounds}")
        click.echo()

    # iterate over the layers
    if not quiet:
        click.echo(f"SEGMENTS")
        click.echo(f"--------")
        click.echo(f"Found {len(layernames)} layers in {segments}")

    # get the variogram parameters
    vario_params = dict(
        model=model,
        n_lags=n_lags,
        maxlag=float(maxlag) if maxlag and maxlag not in ('mean', 'median') else maxlag,
        use_nugget=use_nugget
    )

    # add anything that was passed to the context
    extra = dict()
    try:
        extra = {key.strip('-'): value for key, value in [s.split('=') for s in ctx.args]}
    except ValueError:
        click.echo('WARNING: could not parse all optional parameters. Ignoring them.')

    vario_params.update(extra)

    if not quiet:
        click.echo('\nVARIOGRAM PARAMETERS')
        click.echo('--------------------')
        click.echo(vario_params)

    # check if the CRS will be ignored
    if ignore_crs:
        crs = CRS.from_user_input('+proj=tmerc +datum=WGS84 +units=cm +no_defs')
    else:
        crs = raster.crs

    # process each layer
    for layername in layernames:
        # open the layer
        segment = gpd.read_file(segments, layer=layername)
        
        # either use the local CRS or transform to RASTER crs
        if ignore_crs:
            segment.set_crs(crs, allow_override=True, inplace=True)
        else:
            segment.to_crs(crs, inplace=True)

        if not quiet:
            click.echo(f"\nProcessing layer {layername} containing {len(segment)} zones")

        # this is the actual implementation
        try:
            clips, transforms = clip_features(raster, segment, nomask=False, dtype=dtype, quiet=quiet)
        except Exception as e:
            click.echo(f"ERROR clipping layer {layername}: {str(e)}")
            continue
        
        # do the univariate statistics
        try:
            uni_segments, uni_stats = add_univariate_to_segmentation(clips, segment, add_to_features=not stats_to_file, axis=axis, use_band=use_band)
        except Exception as e:
            click.echo(f"ERROR univariate statistics calculation for layer {layername}: {str(e)}")
            continue

        # do the variograms
        try:
            vario_segments, variograms, parameters = add_variograms_to_segmentation(clips, uni_segments, n=sample, seed=seed, quiet=quiet, **vario_params)
        except Exception as e:
            click.echo(f"ERROR variogram calculation for layer {layername}: {str(e)}")
            continue

        # check if we want to save the univariate statistics to file
        if stats_to_file:
            # stats folder
            stats_folder = os.path.join(os.path.dirname(output_file), layername, 'stats')
            os.makedirs(stats_folder, exist_ok=True)

            # save the stats
            for i, stat in enumerate(uni_stats):
                # first check the dimensions, because it could be broadcasted along two axes
                if stat.ndim != 2:
                    # build the filename
                    stat_file = os.path.join(stats_folder, f"{layername}_stats_{i + 1}.txt")
                    np.savetxt(stat_file, stat)
                else:
                    # we can go for a csv
                    stat_file = os.path.join(stats_folder, f"{layername}_stats_{i + 1}.csv")
                    
                    # handle the case, where the bands were preserved (ie depth, height, time)
                    # this will result as a 5xN array, where the first dimension is the orignal information coded in the bands
                    if stat.shape[1] == 5:
                        stat = stat.T
                    
                    # make the dataframe
                    df = pd.DataFrame(stat, columns=['mean', 'std', 'min', 'max', 'sum'])
                    df.to_csv(stat_file, index=False)

        # if data uri are not added, we create two new folders
        if not skip_img:
            # create a folder for the clips
            clip_folder = os.path.join(os.path.dirname(output_file), layername, 'clips')
            os.makedirs(clip_folder, exist_ok=True)

            # create a folder for the variograms
            variogram_folder = os.path.join(os.path.dirname(output_file), layername, 'variograms')
            os.makedirs(variogram_folder, exist_ok=True)

            # save the clips
            for i, clip in enumerate(clips):
                clip_file = os.path.join(clip_folder, f"{layername}_clip_{i + 1}.png")
                img = get_raster_band(clip, use_band=use_band)
                ax = plt.imshow(img)
                ax.get_figure().savefig(clip_file, dpi=80)
                plt.close()

            # save the variograms
            # TODO, if this is a List of List with more than one, build a new folder structure
            for i, variogram in enumerate(variograms):
                variogram_file = os.path.join(variogram_folder, f"{layername}_variogram_{i + 1}.png")
                ax = variogram.plot()
                plt.gcf().savefig(variogram_file, dpi=80)
                plt.close()

        # If required, save back the clips
        if add_clip:
            # create a output raster folder
            zone_folder = os.path.join(os.path.dirname(output_file), layername, 'zones')
            os.makedirs(zone_folder, exist_ok=True)

            # go for each combination of clip and transform for each zone
            for i, (clip, transform) in enumerate(zip(clips, transforms)):
                # copy the metadata from the original raster
                out_meta = raster.meta
                
                # update the metadata
                out_meta.update({
                    "driver": "GTiff",
                    "height": clip.shape[1],
                    "width": clip.shape[2],
                    "transform": transform
                })

                # write to file
                zone_name = os.path.join(zone_folder, f"{layername}_zone_{i + 1}.tif")
                with rasterio.open(zone_name, "w", **out_meta) as dest:
                    dest.write(clip)

        # finally save the layer back
        if ignore_crs:
            # force into same CRS as raster
            vario_segments.set_crs(raster.crs, inplace=True, allow_override=True)
        
                # create json files if needed
        if add_json:
            # path 
            json_folder = os.path.join(os.path.dirname(output_file), 'json')
            if not os.path.exists(json_folder):
                os.makedirs(json_folder, exist_ok=True)
            vario_segments.to_file(os.path.join(json_folder, f"{layername}.json"), driver='GeoJSON')
        
        # figure out the write mode
        vario_segments.to_file(output_file, driver='GPKG', layer=layername)
