from config import config
from pixel_sampler import PatchDataset, sar_patch_stats
from histogram import PatchDatasetBandHistograms
import pickle

BANDS = [0,1]

RASTER = '/var/data/3/work/sar_veg/s1m_grd_20180411-20180421_vh_vv.tif'
SHAPEFILE = '/var/data/3/work/sar_veg/references/target-reference%d-2018.shp'

class_train_histograms = class_test_histograms = None

class_train_datasets = [(PatchDataset(RASTER, SHAPEFILE % i, config['pixel_sampler']['band_index'], f_patch_stats=sar_patch_stats)
                         .filter(properties={'split': 'train'}, bands=BANDS)) for i in range(6)]

class_test_datasets = [(PatchDataset(RASTER, SHAPEFILE % i, config['pixel_sampler']['band_index'], f_patch_stats=sar_patch_stats)
                        .filter(properties={'split': 'test'}, bands=BANDS)) for i in range(6)]

with open('train.pkl', 'wb') as f:
        pickle.dump(class_train_datasets, f)

with open('test.pkl', 'wb') as f:
        pickle.dump(class_test_datasets, f)
