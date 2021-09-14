# mdpi-remotesensing-1305061
This code is to accompany an article manuscript sent to MDPI Remote Sensing journal with the title "Density Estimates as Object Representations of Agricultural Fields in Remote Sensing".

The role of this repository is to assist in verification of authenticity of research, accompany article text, illustrate the data flow of the proposed framework, and to provide implementation details for the reader, not as code ready to be run in production.
## Note on data availability
Complying with the EU General Data Protection Regulation (GDPR) and a data usage license granted by the Finnish Food Authority, we cannot disclose the original field parcel annotations described in the manuscript. With best effort however, we aim to publish synthetic annotation samples in the near future, sufficient to verify the workings of the pipeline. The software accepts
- .tiff rasters extracted from S1/S2 imagery. Required image channel order is indicated in config.yml .
- delineation polygons as ESRI .shp shapefiles in the same projection as the .tiffs. 

This code is provided as is, with no guarantees or liabilities. Proper operation requires input data in appropriate format, and adjustment of the data paths in the configuration files.

## Hardware & software requirements
- \>= 16GB RAM. Tens of GB of hard disk space recommended.
- Python 3+
- Conda (optional)
- Matlab (for running LGPDE Laplace approximation w/gpstuff)
- [raster data preprocessing: e.g. ESA SNAP and QGIS]

## Installation
E.g.
```conda create -n mdpi-remotesensing-1305061 python=3```
```conda install numpy scikit-learn pandas pyyaml bzip2 pystan rasterio shapely fiona```

## Generating .tiff raster files from Sentinel-1/Sentinel-2 imagery

Construct .tiff files from Sentinel-1 and Sentinel-2 images using e.g. ESA SNAP and QGIS. Refer to config.yml for S2 band export order. S1 intensity band order is VH, VV. Use common projection for .tiff files as for .shp below. We used EPSG3067 (ETRS89/TM35FIN(E,N)).

## Polygon shapefiles 
Polygons in the ESRI .shp shapefiles are required to have the following property fields: parcelID (values: arbitrary string ids), split (values: train/test)

## Extracting the pixel sets per object: 

See make_msi_datasets.sh, make_sar_dataset.py . The code produces a train and a test dataset for SAR and MSI datasets, e.g. those referred to in the pipeline sample configurations:```train_s2_ndti_ndvi.pkl, test_s2_ndti_ndvi.pkl, train_s1_vh_vv.pkl, test_s1_vh_vv.pkl```


## Running the pipeline

Example pipeline configurations are provided for 
- SAR datasets: GKDE/histogram/LGPDE-laplace X RandomForest/SVC/MLP ( ```pipeline_fast__0_3_4__-30_0_-30_0_30bin.yml```)
- MSI/SAR fusion: ```pipeline_msi_sar_fusion_max_balanced_sar30bin_hist.yml```

The pipeline runs the configuration variations listed in the file, producing a result .tsv with a line for each variation containing e.g. classification accuracy output in addition to DE and classifier used plus dataset sampling information e.g. train vs. test split size. Results can be analyzed and summarized using any software capable of reading .csv/.tsv format. 



E.g. ```python pipeline.py -t pipeline_fast__0_3_4__-30_0_-30_0_30bin.yml```
