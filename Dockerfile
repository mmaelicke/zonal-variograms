ARG python_version=3.10
FROM python:${python_version}

# install requirements for tool-specs
RUN pip install json2args==0.6.0

# build the structure
RUN mkdir /in
RUN mkdir /out
RUN mkdir /src
RUN mkdir -p /tool/lib

# Install GDAL which will be used by geopandas
RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y gdal-bin libgdal-dev
RUN pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}')

# copy over the tool and install
COPY zonal_variograms /tool/lib/zonal_variograms
COPY setup.py /tool/lib/setup.py
COPY requirements.txt /tool/lib/requirements.txt
COPY README.md /tool/lib/README.md
WORKDIR /tool/lib
RUN pip install .

