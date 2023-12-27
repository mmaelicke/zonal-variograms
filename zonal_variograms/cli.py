import click
from pathlib import Path

from zonal_variograms.io import load_dataset, load_segments, save_to_disk
from zonal_variograms.main import add_oid_overlay, spread_oid_from_dataset, clip_features_from_dataset


@click.command('clip', context_settings=dict(help_option_names=['-h', '--help'], ignore_unknown_options=True, allow_extra_args=True))
@click.option('--crs', default=None, type=int, help="The EPSG code of the CRS of the input RASTER. If provided, the RASTER CRS will be overwritten.")
@click.option('--oid', default=None, type=str, help="The attribute of SEGMENTS to use as a unique object id. The segementation will be based on that oid. If not set, the row number will be used.")
@click.option('--raster-backend', default='rio', type=click.Choice(['rio', 'xarray']), help="The backend to use for the RASTER. Defaults to rio.")
@click.option('--overlap', default=False, is_flag=True, help="If set, the SEGMENTS may overlap and the clipping is done procedurally. This can take significantly longer and you may want to batch by oid or set the --use-oid flag.")
@click.option('--use-oid', default=None, type=int, multiple=True, help="If set, only the SEGMENTS with the given oid will be clipped. This can be used to batch the clipping by oid.")
@click.option('--save-path', default=None, type=str, help="Set a base path to save the RASTER clips to. If not set, the parent directory of the RASTER will be used.")
@click.argument('raster')
@click.argument('segments')
@click.pass_context
def clip_dataset(ctx, crs, oid, raster_backend, overlap, use_oid, save_path, raster, segments):
    # derive a save path if not given
    if save_path is None:
        save_path = Path(raster).parent
    else:
        save_path = Path(save_path)

    # read the raster
    try:
        raster = load_dataset(raster, crs=crs, raster_backend=raster_backend)
    except Exception as e:
        raise click.ClickException('Error loading RASTER.') from e

    # read the segments
    for layername, segment in load_segments(segments, crs=raster.rio.crs):
        click.echo(f"Using SEGEMENTS layer {layername} for clipping...")
        # add the OID column if needed
        if oid is None:
            segment['oid'] = range(len(segment))
            oid = 'oid'
        
        # clip the features
        if overlap:
            # clip the features procedurally
            clips = clip_features_from_dataset(raster, segment, oid=oid, n_jobs=-1, use_oids=use_oid, quiet=False)
        else:
            # first rasterize the segments and add to the raster
            ds = add_oid_overlay(raster, segment, oid=oid)

            # clip the features
            clips = spread_oid_from_dataset(ds, oid=oid)
        
        click.echo(f"Clipped {len(clips)}, saving... ")
        # finally save the clips
        save_to_disk(save_path, layername=layername, clips=clips, nested=True)
        click.echo("Done.")


# combine all commands into one cli
@click.group(chain=True)
def cli():
    pass
