#!/bin/bash
#Provide config file placed in repo root dir as first arg $1
# Default is a MSI-SAR data fusion experiment using most features of the pipeline. See and try sar_10_to_300.yml for a more bare-bones example. Write your own or modify parameters for a different pipeline config.

PIPELINE_CONFIG=${1:-msi_sar_fusion_6500.yml} 
echo Running pipeline for config $PIPELINE_CONFIG ...   
docker run -e PROJ_LIB=/usr/local/include/proj -v $(pwd)/out:/out -it luotsi/vegcover python -u ./pipeline.py -t $PIPELINE_CONFIG
