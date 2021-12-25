# vegcover-manuscript-12_2021
This code is to accompany the article manuscript "Density Estimates as Representations of Agricultural Fields for Remote Sensing-Based Monitoring of Tillage and Vegetation Cover".

The role of this repository is to assist in verification of authenticity of research, accompany article text, 
illustrate the data flow of the proposed framework, and to provide implementation details for the reader, 
not as code ready to be run in production.
The code is a Docker-enabled version of the code used for our study that is set up to run essential parts of the 
experiments to demonstrate the data flow of the pipeline including realistic data without requiring 
e.g. a Matlab license or high-power computing resources that we used for computing Stan/MCMC results for LGPDE.
This code is provided as is, with no guarantees or liabilities. 
Proper operation requires input data in appropriate format, and adjustment of the data paths in the configuration files.

## Hardware & software requirements (with Docker)
- Linux or MacOS
- 20GB Hard disk space.
- git
- Docker
- bash
- Memory and CPU:

We ran most experiments on Ubuntu 18.04 with an Intel(R) Core(TM) i3-8350K CPU running at 4.00GHz with 48GB of RAM. 
To run configurations enabling LGPDE (not included, available on request) we used the Helsinki University high-power 
computing "Puhti" environment to run Stan MCMC as parallel processes.

Errata: With less resources you might run into parallel timeouts for the default long-running suite of experiments that runs an 
extensive set of variant combinations of sample sizes, density estimators and classifiers, plus intersects SAR-MSI datasets.
The timeout can currently be adjusted only in code by increasing the hard-coded 900s timeout constant 
_TIMEOUT_PARALLEL_S in module pipeline.py . 

Alternatively, please consider the suite on small samples of SAR data configured in sar_10_to_300.yml used in one of the 
experiments of manuscript Section 3.1 - details given below.

## Running example experiments on Docker for the data of our study:
- (assumes docker and git can be run from command line)
- ```git clone https://github.com/luotsi/vegcover-manuscript-12_2021.git```
- ```cd vegcover-manuscript-12_2021```
- ```./run_pipeline_docker.sh``` (defaults to experiment configuration msi_sar_fusion_6500.yml, run will take a few hours)
- or e.g. ```bash ./run_pipeline_docker.sh msi_sar_fusion_6500.yml```
- or ```./run_pipeline_docker.sh sar_10_to_300.yml```

You can also run a shell in the docker image as usual to explore the runtime environment with the command:

- ```docker run -it luotsi/vegcover bash```

## Running the pipeline outside of docker
- Requires git, conda and Python3

- ```git clone https://github.com/luotsi/vegcover-manuscript-12_2021.git```
- ```cd vegcover-manuscript-12_2021```
- ```conda create -n vegcover-manuscript-12_2021 python=3.7```
- ```conda activate vegcover-manuscript-12_2021```
- ```conda install numpy scikit-learn pandas pyyaml bzip2 pystan rasterio shapely fiona```
- in config.yml, adjust pipeline / output_root_dir to a suitable output directory
- run e.g. ```python -u -t sar_10_to_300.yml```

## Generating your own data
The software accepts
- .tiff rasters extracted from S1/S2 imagery. Required image channel order is indicated in config.yml .
- delineation polygons as ESRI .shp shapefiles in the same projection as the .tiffs. 


### Generating .tiff raster files from Sentinel-1/Sentinel-2 imagery

Construct .tiff files from Sentinel-1 and Sentinel-2 images using e.g. ESA SNAP and QGIS. Refer to config.yml for S2 band export order. S1 intensity band order is VH, VV. Use common projection for .tiff files as for .shp below. We used EPSG3067 (ETRS89/TM35FIN(E,N)).

### Polygon shapefiles 
Polygons in the ESRI .shp shapefiles are required to have the following property fields: parcelID (values: arbitrary string ids), split (values: train/test)

### Extracting the pixel sets per object: 

Once you have produced the raster .tifs and annotated .shp shapefiles and adjusted entries in config.yml / pixel_sampler, run make_datasets.sh . The code produces a train and a test dataset for SAR and MSI datasets, e.g. those referred to in the pipeline sample configurations:```train_s2_ndti_ndvi.pkl, test_s2_ndti_ndvi.pkl, train_s1_vh_vv.pkl, test_s1_vh_vv.pkl```
