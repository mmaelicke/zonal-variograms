"""
Provide some basic input / output logic, primarily for the CLI.
They are separated out here, if you would like to use them in your own scripts.

"""
from typing import Union, Optional, Iterator, List, Tuple
from typing_extensions import Literal
import warnings
from pathlib import Path

import rioxarray as rio
import xarray as xr
import geopandas as gpd
import pandas as pd
import fiona
from pyproj import CRS as pjCRS
from rasterio.crs import CRS as rioCRS
from fiona.errors import DriverError



def load_dataset(
    path: str, 
    crs: Optional[int] = None, 
    raster_backend: Union[Literal['rio'], Literal['xarray']] = 'xarray'
) -> xr.Dataset:
    # first try to open the raster
    try:
        if raster_backend == 'rio':
            raster: Union[xr.Dataset, xr.DataArray] = rio.open_rasterio(path, mask_and_scale=True)
        else:
            raster: xr.Dataset = xr.open_dataset(path, mask_and_scale=True, decode_coords=True)
    except Exception as e:
        raise OSError(f"Could not open RASTER {path}")
    
    # check if we overwrite the crs
    if crs is not None:
        raster.rio.write_crs(crs, inplace=True)
    
    return raster


def load_segments(path: str, crs: Optional[Union[int, pjCRS, rioCRS]] = None) -> Iterator[Tuple[str, gpd.GeoDataFrame]]:
    # load the segments
    layernames = [name for name in fiona.listlayers(path) if name != 'layer_styles']
    
    # go for each layer in segments
    for layername in layernames:
        # load the layer
        try:
            # load the layer from the source
            features = gpd.read_file(path, layer=layername)

            # check if we need to transform the crs
            if crs is not None:
                if isinstance(crs, (pjCRS, rioCRS)):
                    features.to_crs(crs, inplace=True)
                else:
                    features.to_crs(pjCRS.from_epsg(crs), inplace=True)
        except DriverError:
            warnings.warn(f"Could not read layer {layername} in SEGMENTS {path}. Skipping layer.")
            continue

        # yield the geodataframe
        yield layername, features


def save_to_disk(
    base_path: Union[str, Path],
    layername: str,
    oids: Optional[Union[List[int], List[str]]] = None,
    clips: Optional[List[xr.Dataset]] = None,
    aggregates: Optional[List[pd.DataFrame]] = None,
    nested: bool = True,
    file_name: Optional[str] = None,
) -> None:
    # get the base path as absolute path
    base_path = Path(base_path).resolve() / layername

    # create the directory path
    base_path.mkdir(parents=True, exist_ok=True)

    # check if all data arguments are None (nothing to do)
    data_container = [clips, aggregates]
    if all([arg is None for arg in data_container]):
        raise AttributeError(f"Nothing to save. You need to provide at least one of the following arguments: ['clips', 'aggregates']")

    # build an array of oids if not provided
    if oids is None:
        # figure out the length needed
        n = max([len(arg) if arg is not None else 0 for arg in data_container])
        oids = [f"OID_{oid}" for oid in range(n)]
    
    # check if the oids are the same length as the data
    for arg in data_container:
        if arg is None:
            continue
        if len(arg) != len(oids):
            name = f"{arg=}".split('=')[0]
            warnings.warn(f"Length of oids ({len(oids)}) does not match length of '{name}' ({len(arg)}).")


    # check if we need to save clips
    if clips is not None:
        # if the oids are present, we use them to build a structure and name files
        # if they are not present, we build them from a range of integer
        if oids is None:
            oids = [f"OID_{oid}" for oid in range(len(clips))]
        
        # go for each oid and clip
        for oid, clip in zip(oids, clips):
            # build the path
            if nested:
                fpath = base_path / oid
            else:
                fpath = base_path
            fpath.mkdir(parents=True, exist_ok=True)

            # figure out the file name
            if file_name is None:
                ext = '.tif' if len(clip.sizes) == 2 else '.nc'
                file_name = f"{layername}_clip_{oid}{ext}"
            elif '*' in file_name:
                file_name = file_name.replace('*', oid)
            else:
                file_name = f"clip_{oid}_" + file_name
            
            # save the clip
            if file_name.endswith('.nc'):
                clip.to_netcdf(str(fpath / file_name))
            else:
                clip.rio.to_raster(str(fpath / file_name))

    # save the segments
    if aggregates is not None:
        # if the oids are present, we use them to build a structure and name files
        # if they are not present, we build them from a range of integer
        if oids is None:
            oids = [f"OID_{oid}" for oid in range(len(aggregates))]

        # go for each oid and aggregate
        for oid, agg in zip(oids, aggregates):
            # build the path
            if nested:
                fpath = base_path / oid
            else:
                fpath = base_path
            fpath.mkdir(parents=True, exist_ok=True)

            # figure out the file name
            if file_name is None:
                file_name = f"{layername}_aggs_{oid}.parquet"
            elif '*' in file_name:
                file_name = file_name.replace('*', oid)
            else:
                file_name = f"aggs_{oid}" + file_name
            
            # build the file name
            file_name = fpath / file_name
            
            # check if the file already exists
            if file_name.exists():
                # read the file
                if file_name.endswith('parquet'):
                    old = pd.read_parquet(file_name)
                elif file_name.endswith('csv'):
                    old = pd.read_csv(file_name)
                else:
                    old = None
                
                # concat the dataframes
                if old is not None:
                    agg = pd.concat([old, agg], axis=0)
            
            # save the file
            if file_name.endswith('parquet'):
                agg.to_parquet(file_name)
            else:
                agg.to_csv(file_name)
