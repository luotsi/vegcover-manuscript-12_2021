CONFIG_FILE_NAME=sampler_config_msi.yml
DATA_ROOT=/var/data/3/work/sar_veg
RASTER_TIF_PATH=$DATA_ROOT/subset_S2B_MSIL2A_20180418T095029_N0207_R079_T34VFN_20180418T111406_rgb+b12+cld+snw+crci+ndvi_ETRS-T35FIN.tif
OUTPUT_DIR=$DATA_ROOT/msi_shp_parts

for split in 'train' 'test'; do \
  for class_ix in 0 1 2 3 4 5; do \
    SHP_FILE_PATH=$DATA_ROOT/references/target-reference${class_ix}-2018_${split}.shp
    CONFIG=$CONFIG_FILE_NAME python pixel_sampler.py -s $RASTER_TIF_PATH  -a $SHP_FILE_PATH -d $OUTPUT_DIR -o ${split}_${class_ix}.pkl -f msi_patch_stats
  done
done