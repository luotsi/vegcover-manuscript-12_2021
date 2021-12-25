#!/bin/bash
#Provide config file placed in repo root dir as first arg $1
PIPELINE_CONFIG=$1
if "$PIPELINE_CONFIG" == "":
   PIPELINE_CONFIG=sar_10_to_300.yml # default 
fi
   
docker run -v $(pwd)/out:/out -it vegcover bash ./pipeline.py -t sar_10_to_300.yml; chmod o+rw /out
