ARG python_version=3.10.13
FROM python:${python_version}

# install requirements for tool-specs
RUN pip install json2args==0.6.1

# build the structure
RUN mkdir /in
RUN mkdir /out
RUN mkdir /src
RUN mkdir -p /tool/lib

# Install GDAL which will be used by geopandas
RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y gdal-bin libgdal-dev
RUN pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}')

# install all dependencies with exact versions
RUN pip install geopandas==0.14.1
RUN pip install geocube==0.4.2
RUN pip install rasterio==1.3.9
RUN pip install rioxarray==0.15.0
RUN pip install cftime==1.6.3
RUN pip install click==8.1.7
RUN pip install scikit-gstat==1.0.12
RUN pip install typing_extensions==4.9.0
RUN pip install xarray==2023.12.0
RUN pip install scipy==1.11.4
RUN pip install joblib==1.3.2

# copy over the tool
COPY zonal_variograms /tool/lib/zonal_variograms
COPY setup.py /tool/lib/setup.py
COPY requirements.txt /tool/lib/requirements.txt
COPY README.md /tool/lib/README.md

# install the tool
WORKDIR /tool/lib
RUN pip install -r requirements.txt
RUN pip install .

# do the tool-spec specific setup
COPY src /src
COPY in /in
WORKDIR /src

CMD ["python", "run.py"]

