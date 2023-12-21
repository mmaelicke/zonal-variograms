from typing import List, Union, Optional, Tuple
from typing_extensions import Literal

import rasterio
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import skgstat as skg
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt

from zonal_variograms.util import mpl_to_base64


def clip_features(raster: rasterio.DatasetReader, features: gpd.GeoDataFrame, nomask: bool = False, dtype: str = None, quiet: bool = False) -> Tuple[List[np.ndarray], list]:
    """
    Clips the raster dataset using the geometries from the GeoDataFrame.

    Parameters
    ----------
    raster : rasterio.DatasetReader
        The raster dataset to be clipped.
    features : gpd.GeoDataFrame
        The GeoDataFrame containing the geometries for clipping.
    nomask bool, optional
        If set to `False` (default), the output arrays will be masked numpy 
        arrays, masked yb the raster nodata value. If set to `True`, an ordinary
        numpy array will be returned.
    dtype : str, optional
        The data type of the output array, as respresented by the data, not the actual used
        data type of the source. If this is not `None`, the specified dtype will be used to
        unpack the data values in a unsafe way, by scaling and offsetting it with the 
        scale and offset as specified in the raster metadata.
        This is frequently used in large raster files (ERA5-Land) to save space (by storing
        2 byte values, which are shifted to a float32).
        Defaults to `None`. 
    quiet : bool, optional
        Whether to display progress bar. Defaults to False.

    Returns
    -------
    List[np.ndarray]
        A list of clipped arrays.
    """
    # function implementation...
    # build an iterator
    if quiet:
        _iterator = features.geometry
    else:
        _iterator = tqdm(features.geometry)
    
    # create result containers
    clipped_arrays = []
    clipped_transforms = []

    for geometry in _iterator:
        # clip the feature
        clipped_array, clipped_transform = mask(raster, [geometry], crop=True)
        
        # append the results
        clipped_transforms.append(clipped_transform)

        # check if we need to mask the array
        if not nomask:
            clipped_array = np.ma.masked_values(clipped_array, raster.nodata)

        # check if the value space needs to be shifted
            # for some sources (ie. ERA5) we need to unpack the 2byte data into floats by using scale and offset raster metadata
        if dtype is not None:
            clipped_array = np.add(np.multiply(clipped_array.astype(dtype, casting="unsafe"), raster.scales[0], casting="unsafe"), raster.offsets[0], casting="unsafe")

        # append the array
        clipped_arrays.append(clipped_array)

    # return features
    return clipped_arrays, clipped_transforms


def get_raster_band(arr: np.ndarray, use_band: int = 0) -> np.ndarray:
    """
    Extract a single band at index `use_band` from a raster array.
    """
    # check if the array is 3D
    if len(arr.shape) == 3:
        # get the correct band
        return arr[use_band]
    elif len(arr.shape) > 3:
        raise ValueError(f'Array has more than 3 dimensions. Got {len(arr.shape)} dimensions. Sorry cannot handle that.')
    else:
        return arr


def raster_variogram(raster: np.ndarray, **vario_params) -> skg.Variogram:
    """
    Calculates the variogram for a raster dataset.

    Parameters
    ----------
    raster : np.ndarray
        The raster dataset.
    **vario_params : dict
        Additional parameters for the skgstat.Variogram class.

    Returns
    -------
    skg.Variogram
        The calculated variogram.
    """
    # function implementation...
    # span a meshgrid over both axes
    x, y = np.meshgrid(np.arange(raster.shape[1]), np.arange(raster.shape[0]))

    # stack into a coordinate array
    coords = np.stack([x.flatten(), y.flatten()], axis=-1)

    # get the values from the raster
    z = raster.flatten()

    # calculate the variogram
    return skg.Variogram(coords, z, **vario_params)


def raster_sample_variogram(raster: np.ndarray, n: int = 1000, seed: int = 1312, **vario_params) -> skg.Variogram:
    """
    Calculates the sample variogram for a raster dataset.

    Parameters
    ----------
    raster : np.ndarray
        The raster dataset.
    n : int, optional
        The number of samples to use for calculating the variogram. Defaults to 1000.
    seed : int, optional
        The seed for the random number generator. Defaults to 1312.
    **vario_params : dict
        Additional parameters for the skgstat.Variogram class.

    Returns
    -------
    skg.Variogram
        The sample variogram.
    """
    # function implementation...
    # span a meshgrid over both axes
    x, y = np.meshgrid(np.arange(raster.shape[1]), np.arange(raster.shape[0]))

    # stack into a coordinate array
    coords = np.stack([x.flatten(), y.flatten()], axis=-1)

    # get the values from the raster
    z = raster.flatten()

    # build an index over the values
    idx = np.arange(len(z))

    # shuffle the idx in place using a seeded rng
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    # calculate the variogram on the n first shuffled values
    return skg.Variogram(coords[idx[:n]], z[idx[:n]], **vario_params)


def estimate_empirical_variogram(
    raster: Union[np.ndarray, List[np.ndarray]],
    n: Optional[int] = 1000,
    seed: int = 1312,
    quiet: bool = False,
    use_band: Optional[Union[Literal['all'], int]] = None,
    np_agg = np.mean,
    **vario_params
) -> List[List[skg.Variogram]]:
    """
    Estimates the empirical variogram for one or multiple raster datasets.

    Parameters
    ----------
    raster : Union[np.ndarray, List[np.ndarray]]
        The raster dataset(s).
    n : int, optional
        The number of samples to use for calculating the variogram. Defaults to 1000.
    seed : int, optional
        The seed for the random number generator. Defaults to 1312.
    quiet : bool, optional
        Whether to display progress bar. Defaults to False.
    use_band : int, optional
        The band to use from the raster dataset. Defaults to 0.
    **vario_params : dict
        Additional parameters for the skgstat.Variogram class.

    Returns
    -------
    List[skg.Variogram]
        A list of empirical variograms.
    """
    # check the type of the input raster
    if isinstance(raster, np.ndarray):
        raster = [raster]

    # determine the correct variogram function
    if n is None:
        vario_func = raster_variogram
    else:
        vario_func = lambda arr: raster_sample_variogram(arr, n=n, seed=seed, **vario_params)
    
    # output container
    variograms = []

    # build and iterator
    if quiet:
        _iterator = raster
    else:
        _iterator = tqdm(raster)
    
    # go for each band in the raster
    for arr in _iterator:
        # only use the first band or spread the across all bands
        if use_band is None:
            # aggregate along the first axis, if dimension is larger than 2
            if arr.ndim > 2:
                arr = np_agg(arr, axis=0)
            variograms.append([vario_func(arr)])
        
        # we use only a single band
        if isinstance(use_band, int):
            variograms.append([vario_func(get_raster_band(arr, use_band=use_band))])
        
        if isinstance(use_band, str) and use_band == 'all':
            # always just iterate along the axis 0 and create a variogram for each slice
            variograms.append([vario_func(get_raster_band(arr, use_band=i)) for i in range(arr.shape[0])])

        else:
            raise AttributeError(f"Invalid value for `use_band`. Got {use_band}. Must be either `None`, `int`, or `'all'`.")
    
    # return the variograms
    return variograms


def univariate_statistics(arr: Union[np.ndarray, List[np.ndarray]], axis: Optional[int] = None, use_band: Optional[int] = None) -> List[np.ndarray]:
    """
    Calculates the univariate statistics for one or multiple raster datasets.
    """
    # check if only one array was given
    if isinstance(arr, np.ndarray):
        arr = [arr]
    
    # create a result container
    # results = np.ones((len(arr), 5), dtype=float) * np.nan
    results = []

    # go for each polygon
    for i, layer in enumerate(arr):
        # only use the first band or spread the across all bands
        if use_band is not None:
            layer = get_raster_band(layer, use_band=use_band)

        # calculate the statistics
        results.append(np.asarray([
            layer.mean(axis=axis),
            layer.std(axis=axis),
            layer.min(axis=axis),
            layer.max(axis=axis),
            layer.sum(axis=axis)
        ]))
    
    # return the results
    return results


def add_univariate_to_segmentation(
    clipped_arrays: Union[np.ndarray, List[np.ndarray]],
    features: gpd.GeoDataFrame,
    add_to_features: bool = True,
    inplace: bool = False,
    use_band: Optional[Union[Literal['all'], int]] = None,
) -> Tuple[gpd.GeoDataFrame, List[np.ndarray]]:
    """"""
    # reduce each slice into a single value or reduce along the first axis
    # first calculate the univariate statistics
    
    # aggregate to univariate distribution moments along all axes
    if use_band is None:
        stats = univariate_statistics(clipped_arrays, axis=None, use_band=None)
    
    # aggregate to univariate distribution moments for only one band
    elif isinstance(use_band, int):
        stats = univariate_statistics(clipped_arrays, axis=None, use_band=use_band)
    
    # go for all bands and reduce to series of moments along the first axis
    elif isinstance(use_band, str) and use_band == 'all':
        # only preserve the first axis
        axis = tuple([dim for dim in range(clipped_arrays[0].ndim) if dim != 0])
        stats = univariate_statistics(clipped_arrays, axis=axis, use_band=None)
    
    # TODO: here we could add spatially distributed statistical moment (mean in each cell etc)

    else:
        raise AttributeError(f"Invalid value for `use_band`. Got {use_band}. Must be either `None`, `int`, or `'all'`.")

    # get the standard names
    colnames = ['mean', 'std', 'min', 'max', 'sum']
    
    # check if we are forced to not add to features
    if not add_to_features:
        return features, stats
    
    # check if there are too many dimensions on the result
    if np.asarray(stats[0]).size != 5:
        # if we spread dimensions, do it and add, if not, skip the adding part
        if stats[0][0].size % 5 == 0:
            # get the output dimension
            out_dim = stats[0][0].shape[0]

            # reorder the data
            stats = [np.asarray(row).resize(out_dim * 5, 1).flatten() for row in stats]
            colnames = [f"{name}_{i + 1}" for name in colnames for i in range(out_dim)]

        else:
            # just return
            return features, stats

    # create the dataframe
    statistics = pd.DataFrame(data=stats, columns=colnames)

    # copy the input data if not inplace
    if not inplace:
        segments = features.copy()
    else:
        segments = features
    
    # add the parameters to the segments
    segments = segments.join(statistics)

    # finally return everything
    return segments, stats


def add_variograms_to_segmentation(
    clipped_arrays: Union[np.ndarray, List[np.ndarray]],
    features: gpd.GeoDataFrame,
    add_to_features: bool = True,
    n: Optional[int] = 1000,
    seed: int = 1312,
    quiet: bool = False,
    inplace: bool = False,
    use_band: Optional[Union[Literal['all'], int]] = None,
    **vario_params
) -> Tuple[gpd.GeoDataFrame, List[List[skg.Variogram]], List[np.ndarray]]:
    """
    Adds variogram parameters to a segmentation geopackage.

    Parameters
    ----------
    clipped_arrays : Union[np.ndarray, List[np.ndarray]]
        The clipped raster arrays.
    features : gpd.GeoDataFrame
        The segmentation layer.
    add_to_features : bool, optional
        Whether to add the variogram parameters to the features. Defaults to True.
    n : int, optional
        The number of samples to use for calculating the variogram. Defaults to 1000. 
        If None, all data will be used. Caution: that can turn out to be a lot of data.
    seed : int, optional
        The seed for the random number generator. Defaults to 1312.
    quiet : bool, optional
        Whether to display progress bar. Defaults to False.
    inplace : bool, optional
        Whether to modify the input GeoDataFrame inplace. Defaults to False.
    use_band : int, 'all', optional
        The band to use from the raster dataset. Defaults to None.
    **vario_params : dict
        Additional parameters for the skgstat.Variogram class.

    Returns
    -------
    Tuple[gpd.GeoDataFrame, List[List[skg.Variogram]], List[np.ndarray]]
        The updated GeoDataFrame, the list of variograms and the list of the variogram parameters.

    """
    # the calculate the variograms
    features_variograms = estimate_empirical_variogram(clipped_arrays, n=n, seed=seed, quiet=quiet, use_band=use_band, **vario_params)

    # Each feature can either have one, or many variograms (for all bands)
    if isinstance(use_band, str) and use_band == 'all':
        # handle multiple variograms
        parameters = []
        for variograms in features_variograms:
            # get the parameters
            param = np.asarray([v.parameters for v in variograms])
            colnames = ['vario_range', 'vario_sill', 'vario_nugget'] if param.shape[1] == 3 else ['vario_range', 'vario_sill', 'vario_shape', 'vario_nugget']
            params = pd.DataFrame(data=param, columns=colnames)
            
            # add nugget to sill ratio
            params['nugget_sill_ratio'] = (params.vario_nugget / (params.vario_sill + params.vario_nugget)).round(2)

            # add the r2 for each variogram
            params['vario_r2'] = [v.r2 for v in variograms]
            
            parameters.append(params)
        
        # return 
        return features, features_variograms, parameters
    else:
        # flatten the list of variograms
        variograms = [vars[0] for vars in features_variograms]

        # add variogram parameters to the features
        params = np.asarray([v.parameters for v in variograms])
        colnames = ['vario_range', 'vario_sill', 'vario_nugget'] if params.shape[1] == 3 else ['vario_range', 'vario_sill', 'vario_shape', 'vario_nugget']
        parameters = pd.DataFrame(data=params, columns=colnames)
        
        # add nugget to sill ratio
        parameters['nugget_sill_ratio'] = (parameters.vario_nugget / (parameters.vario_sill + parameters.vario_nugget)).round(2)

        # add the r2 for each variogram
        parameters['vario_r2'] = [v.r2 for v in variograms]

        # copy the input data if not inplace
        if not inplace:
            segments = features.copy()
        else:
            segments = features
        
        # turn add the parameters to the segments
        segments = segments.join(parameters)

        # finally return everything
        return segments, features_variograms, parameters
