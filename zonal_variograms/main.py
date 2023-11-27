from typing import List, Union, Optional, Tuple
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


def clip_features(raster: rasterio.DatasetReader, features: gpd.GeoDataFrame, quiet: bool = False) -> Tuple[List[np.ndarray], list]:
    """
    Clips the raster dataset using the geometries from the GeoDataFrame.

    Parameters
    ----------
    raster : rasterio.DatasetReader
        The raster dataset to be clipped.
    features : gpd.GeoDataFrame
        The GeoDataFrame containing the geometries for clipping.
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
        clipped_arrays.append(clipped_array)
        clipped_transforms.append(clipped_transform)

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
    use_band: int = 0,
    **vario_params
) -> List[skg.Variogram]:
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
    # function implementation...
    # determine the correct variogram function
    if n is None:
        vario_func = raster_variogram
    else:
        vario_func = lambda arr: raster_sample_variogram(arr, n=n, seed=seed, **vario_params)
    
    # check if more than one raster is given
    if isinstance(raster, (list, tuple)):
        # iterate over the rasters
        if quiet:
            return [vario_func(get_raster_band(arr, use_band=use_band)) for arr in raster]
        else:
            return [vario_func(get_raster_band(arr, use_band=use_band)) for arr in tqdm(raster)]
    else:
        return vario_func(get_raster_band(raster, use_band=use_band))


def add_variograms_to_segmentation(
    raster: rasterio.DatasetReader,
    features: gpd.GeoDataFrame,
    n: Optional[int] = 1000,
    seed: int = 1312,
    quiet: bool = False,
    inplace: bool = False,
    add_data_uri: bool = False,
    use_band: int = 0,
    **vario_params
) -> Tuple[gpd.GeoDataFrame, List[np.ndarray], list, List[skg.Variogram]]:
    """
    Adds variogram parameters to a segmentation geopackage.

    Parameters
    ----------
    raster : rasterio.DatasetReader
        The raster dataset.
    features : gpd.GeoDataFrame
        The segmentation layer.
    n : int, optional
        The number of samples to use for calculating the variogram. Defaults to 1000. 
        If None, all data will be used. Caution: that can turn out to be a lot of data.
    seed : int, optional
        The seed for the random number generator. Defaults to 1312.
    quiet : bool, optional
        Whether to display progress bar. Defaults to False.
    inplace : bool, optional
        Whether to modify the input GeoDataFrame inplace. Defaults to False.
    add_data_uri : bool, optional
        Whether to add data uris for the variograms and cropped images. Defaults to False.
    use_band : int, optional
        The band to use from the raster dataset. Defaults to 0.
    **vario_params : dict
        Additional parameters for the skgstat.Variogram class.

    Returns
    -------
    Tuple[gpd.GeoDataFrame, List[np.ndarray], List[skg.Variogram]]
        The updated GeoDataFrame, the list of cropped arrays, and the list of variograms.

    """
    # first, clip the raster features
    clipped_arrays, clipped_transforms = clip_features(raster, features, quiet=quiet)

    # the calculate the variograms
    variograms = estimate_empirical_variogram(clipped_arrays, n=n, seed=seed, quiet=quiet, **vario_params)

    # add variogram parameters to the features
    params = np.asarray([v.parameters for v in variograms])
    colnames = ['vario_range', 'vario_sill', 'vario_nugget'] if params.shape[1] == 3 else ['vario_range', 'vario_sill', 'vario_shape', 'vario_nugget']
    parameters = pd.DataFrame(data=params, columns=colnames)
    
    # add nugget to sill ratio
    parameters['nugget_sill_ratio'] = (parameters.vario_nugget / (parameters.vario_sill + parameters.vario_nugget)).round(2)

    # add data uris if needed
    if add_data_uri:
        # define a local function to plot
        def _plot_v(vario):
            fig = vario.plot().figure
            uri = mpl_to_base64(fig, as_data_uri=True)
            plt.close(fig)
            return uri
        
        # plot the cropped image
        def _plot_c(arr):
            img = get_raster_band(arr, use_band=use_band)
            fig = plt.imshow(img).get_figure()
            plt.tight_layout()
            uri = mpl_to_base64(fig, as_data_uri=True)
            plt.close(fig)
            return uri

        # map the helper function to the variograms and save results
        parameters['variogram_data_uri'] = list(map(_plot_v, variograms))
        parameters['crop_data_uri'] = list(map(_plot_c, clipped_arrays))

    # copy the input data if not inplace
    if not inplace:
        segments = features.copy()
    else:
        segments = features
    
    # turn add the parameters to the segments
    segments = segments.join(parameters)

    # finally return everything
    return segments, clipped_arrays, clipped_transforms, variograms
