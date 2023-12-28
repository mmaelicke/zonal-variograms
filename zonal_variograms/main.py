from typing import List, Tuple, Optional, Union
import warnings

from tqdm import tqdm
from xarray import Dataset
import geopandas as gpd
import pandas as pd
import numpy as np
from geocube.api.core import make_geocube
from rasterio.errors import ShapeSkipWarning
from joblib import delayed, Parallel


def add_oid_overlay(raster: Dataset, features: gpd.GeoDataFrame, oid: str = 'oid') -> Dataset:
    # make sure the Dataset has the rioxarray extension installed
    if not hasattr(raster, 'rio'):
        raise ValueError('raster Dataset must be loaded with rioxarray extension installed. Install with `pip install rioxarray`')
    
    # right now, oid literal cannot be changed
    if oid != 'oid':
        raise NotImplementedError("oid has to be the literal 'oid' and cannot be changed. This will change with a future version")
    
    # check if the features have an id column
    if 'oid' not in features.columns:
        features['oid'] = range(len(features))

    # rasterize the features
    cube = make_geocube(
        vector_data=features,
        measurements=[oid],
        like=raster,
        interpolate_na_method='cubic'
    )

    # add all the variables from the raster to the cube
    for var in raster.data_vars:
        cube[var] = raster[var]
    
    # return the cube
    return cube


def spread_oid_from_dataset(cube: Dataset, oid: str = 'oid') -> List[Dataset]:
    # right now, oid literal cannot be changed
    if oid != 'oid':
        raise NotImplementedError("oid has to be the literal 'oid' and cannot be changed. This will change with a future version")

    # get a list of all unique ids
    clip_datasets = []
    # go for each unique id
    for oid in np.unique(cube.oid.data):
        # crop the cube
        cube_slice = cube.where(cube.oid == oid)

        # dropna along all axes
        clip = None
        for dim in cube_slice.dims.keys():
            if clip is None:
                clip = cube_slice.dropna(dim, how='all')
            else:
                clip = clip.dropna(dim, how='all')
        
        # append the cropped cube to the list
        clip_datasets.append(clip)
    
    # return the list
    return clip_datasets


def clip_features_from_dataset(
    raster: Dataset,
    features: gpd.GeoDataFrame, 
    oid: str = 'oid', 
    use_oids: Optional[Union[int, List[int]]] = None, 
    n_jobs: Optional[int] = None,
    quiet: bool = False
) -> List[Dataset]:
    # make sure the Dataset has the rioxarray extension installed
    if not hasattr(raster, 'rio'):
        raise ValueError('raster Dataset must be loaded with rioxarray extension installed. Install with `pip install rioxarray`')
    
    # right now, oid literal cannot be changed
    if oid != 'oid':
        raise NotImplementedError("oid has to be the literal 'oid' and cannot be changed. This will change with a future version")

    # check if the features have an id column
    if oid not in features.columns:
        features[oid] = range(len(features))

    # build a handler function to clip the featues at one index
    def _handler(_oid: Union[int, float]) -> Dataset:
        geom = features.where(features[oid] == _oid).dropna().geometry.values.tolist()
        clip = raster.copy().rio.clip(geom, features.crs, drop=True, invert=False)

        # make a geocube of only this feature
        with warnings.catch_warnings():
            # we expect skipped shapes here
            warnings.simplefilter("ignore", category=ShapeSkipWarning)
            
            # make a new geocube with the catchment in it
            cube = make_geocube(features.where(features[oid] == _oid), measurements=[oid], like=clip, interpolate_na_method='cubic')

        # add all variables to the cube
        for var in clip.data_vars:
            cube[var] = clip[var]

        # return the cube
        return cube

    
    # set up the input parameters
    if use_oids is not None:
        if not isinstance(use_oids, (list, tuple, np.ndarray)):
            use_oids = [use_oids]

        inputs = [_oid for _oid in features[oid].values if _oid in use_oids]
    else:
        inputs = features[oid].values

    # check if we need to parallelize
    if n_jobs is not None:
        # build the worker and delayed function
        worker = Parallel(n_jobs=n_jobs, return_as='list' if quiet else 'generator')
        delayed_handler = delayed(_handler)
        # build the iterator
        if quiet:
            return list(worker(delayed_handler(_oid) for _oid in inputs))
        else:
            return list(tqdm(worker(delayed_handler(_oid) for _oid in inputs)))
    
    else:
        if quiet:
            return [_handler(_oid) for _oid in inputs]
        else:
            return [_handler(_oid) for _oid in tqdm(inputs)]


def univariate_by_oid(dataset: Dataset,  oid: str = 'oid') -> pd.DataFrame:
    # right now, oid literal cannot be changed
    if oid != 'oid':
        raise NotImplementedError("oid has to be the literal 'oid' and cannot be changed. This will change with a future version")

    # group the dataset
    grouped = dataset.groupby('oid')

    # create the base grouping
    aggregates = None
    for agg_name in ('mean', 'std', 'min', 'max', 'median', 'sum'):
        # aggregate using the given functions
        aggregator = getattr(grouped, agg_name)
        agg_df = aggregator().to_dataframe().drop('spatial_ref', axis=1, errors='ignore')

        # rename the columns
        agg_df.columns = [f'{col}_{agg_name}' for col in agg_df.columns]

        # check if we need to add to data or create a new one
        if aggregates is None:
            aggregates = agg_df.copy()
        else:
            for col in agg_df.columns:
                aggregates[col] = agg_df[col]

    # return 
    return aggregates


def add_aggregates_to_segmentation(aggregates: pd.DataFrame, segments: gpd.GeoDataFrame, oid: str = 'oid') -> Tuple[gpd.GeoDataFrame, List[pd.DataFrame]]:
    # right now, oid literal cannot be changed
    if oid != 'oid':
        raise NotImplementedError("oid has to be the literal 'oid' and cannot be changed. This will change with a future version")
    
    # container for the output dataframes
    output_dataframes = []

    # aggregate the file over the oids to access the single groups
    for oid, df in aggregates.groupby(oid):
        # first reset the index and drop the oid column
        data = df.reset_index().drop(oid, axis=1).copy()

        # append to output
        output_dataframes.append(data)

    # if there is only exactly one value for each oid, we can
    # directly join that to the segments
    if aggregates.shape[0] == segments.shape[0]:
        new_segments = segments.join(aggregates, on=oid)
    else:
        # build the aggregation dictionary use the suffix or mean
        aggs = {col: col.split('_')[-1] if '_' in col or '_std' in col else 'mean' for col in df.columns}

        # further aggregate the aggregates dict and joint to segments
        new_segments = segments.join(df.groupby(oid).agg(aggs))
        
    # return the new structures
    return new_segments, output_dataframes
