# Zonal variograms

This small piece of code can generate variograms using a zonal statistics approach. For any given geoTiff, it will calculate a variogram for any Polygon supplied to tool.
It accepts shapefiles and geopackages, and can handle multiple layers as input. The output is always a geopackage with the same inputs. Zonal variogram parameters are
added as columns to the properties table.
By default, the tool will calculate a variogram for all input cells, but can be configured to rather use a random subsample. This can be helpful in case the zones are big.

The variograms will share the same hyper-parameters, thus differences are only due to different variogram parameters fitted to the data. Multiple parameters are not possible right not.
The tool does not support multi-band input right now.

The tool can be used as a Python library or a command line tool

Install like:
```
pip install zonal_variograms
```

Example call
```bash
zonal_variograms ./in/my_raster.tif ./in/Features.gpkg   --model=exponential --maxlag=median --use-nugget --n-lags=25 --sample=400 --add-json --add-data-uri
```
