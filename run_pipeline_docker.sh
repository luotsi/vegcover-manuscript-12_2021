#!/bin/bash
#Provide config file placed in repo root dir as first arg $1
# Default is the SAR experiment. See and try msi_sar_fusion_6500.yml for a more advanced example. Write your own or modify parameters for a different pipeline config.

PIPELINE_CONFIG=${1:-sar_10_to_300.yml} 
echo Running pipeline for config $PIPELINE_CONFIG ...   
docker run -v $(pwd)/out:/out -it luotsi/vegcover python ./pipeline.py -t sar_10_to_300.yml; chown -R $(id -u) /out
