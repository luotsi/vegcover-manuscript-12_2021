import pickle
import time

import numpy as np
import pandas as pd

from config import config
import pixel_sampler as pxs
import functools as ft
import operator as op
from glob import glob


CLOUD_THRESHOLD_PCT = 10
SNOW_THRESHOLD_PCT = 10


def valid_density_dataset(class_densities, class_auxiliary_data):
    vd_ix = _valid_density_index(class_densities)
    class_densities = [d[vd_ix[class_ix], :] for class_ix, d in enumerate(class_densities)]
    class_patch_ids = []
    class_patch_stats =[]

    for class_ix, auxiliary_data in enumerate(class_auxiliary_data[0]):
        patch_ids = np.array(auxiliary_data)
        patch_ids = patch_ids[vd_ix[class_ix]].tolist()
        class_patch_ids.append(patch_ids)

    for class_ix, auxiliary_data in enumerate(class_auxiliary_data[1]):
        patch_stats = np.array(auxiliary_data)
        patch_stats = patch_stats[vd_ix[class_ix]].tolist()
        class_patch_stats.append(patch_stats)

    return class_densities, (class_patch_ids, class_patch_stats)


def _patch_px_limiter(n):
    def limiter(patch):
        return (np.sum(patch.band_pixels) != 0) and (n is None or patch.band_pixels.shape[0] <= n)
    return limiter


def _extract_patch_ids(dataset_classes):
    return [[patch.patch_properties['parcelID'] for patch in clazz.patch_data] for clazz in dataset_classes]


def _extract_patch_stats(dataset_classes):
    return [[patch.patch_stats for patch in class_dataset.patch_data] for class_dataset in dataset_classes]


def _valid_density_index(class_densities):
    class_valid_ix = [np.logical_not(np.any(np.isnan(class_densities[class_ix]), axis=1))
                      for class_ix in range(len(class_densities))]
    return class_valid_ix


def _msi_cloud_and_snow_filter(pixels):
    rev_band_index = pxs.reverse_band_index(pxs.DEFAULT_MSI_BAND_INDEX)
    cloud_threshold = pixels[:, rev_band_index['p_cld']] <= CLOUD_THRESHOLD_PCT
    snow_threshold = pixels[:, rev_band_index['p_snw']] <= SNOW_THRESHOLD_PCT
    return pixels[np.logical_and(cloud_threshold, snow_threshold)]


_PX_FILTERS = {'msi_cloud_and_snow_filter': _msi_cloud_and_snow_filter}

class PatchIntersection:
    @staticmethod
    def intersection(cfg: dict, _):
        n_train_samples = cfg['n_train_samples']
        n_test_samples = cfg['n_test_samples']
        included_original_classes = cfg['included_original_classes']
        dataset_ids = []
        for ds in cfg['datasets']:
            px_filter = _PX_FILTERS[ds['px_filter']] if 'px_filter' in ds else lambda x: x
            bands = ds['included_bands'] if 'included_bands' in ds else None
            train_datasets = pxs.PatchDataset.load(ds['train_path'])
            test_datasets = pxs.PatchDataset.load(ds['test_path'])
            train_ids = [[patch.patch_properties['parcelID']
                          for patch in class_train_ds.filter(bands=bands,
                                                             px_filter=px_filter).patch_data
                          if patch.band_pixels.shape[0] > pxs.MIN_NUM_PIXELS_PER_PATCH and np.sum(patch.band_pixels) != 0]
                         for class_train_ds in train_datasets]
            test_ids = [[patch.patch_properties['parcelID']
                         for patch in class_test_ds.filter(bands=bands,
                                                           px_filter=px_filter).patch_data
                         if patch.band_pixels.shape[0] > pxs.MIN_NUM_PIXELS_PER_PATCH and np.sum(patch.band_pixels) != 0]
                        for class_test_ds in test_datasets]
            dataset_ids.append(dict(train=train_ids, test=test_ids))
        train_ids_per_class = list(zip(*[ids['train'] for ids in dataset_ids]))
        test_ids_per_class = list(zip(*[ids['test'] for ids in dataset_ids]))
        train_intersection_per_class = [list(ft.reduce(lambda acc, e: acc.intersection(e),
                                                    train_ids_per_dataset[1:],
                                                    set(train_ids_per_dataset[0])))
                                        for train_ids_per_dataset in train_ids_per_class]
        test_intersection_per_class = [list(ft.reduce(lambda acc, e: acc.intersection(e),
                                                   test_ids_per_dataset[1:],
                                                   set(test_ids_per_dataset[0])))
                                       for test_ids_per_dataset in test_ids_per_class]

        return dict(train=[np.random.choice(train_intersection, len(train_intersection) if n_train_samples is None else n_train_samples, replace=False).tolist()
                           if class_ix in included_original_classes else []
                           for class_ix, train_intersection in enumerate(train_intersection_per_class)],
                    test=[np.random.choice(test_intersection, len(test_intersection) if n_test_samples is None else n_test_samples, replace=False).tolist()
                          if class_ix in included_original_classes else []
                          for class_ix, test_intersection in enumerate(test_intersection_per_class)])


class DatasetReaders():
    @staticmethod
    def patch_dataset_deferred(cfg, input_):
        deferring_output_paths = cfg['deferring_output_paths']
        expanded_paths = []
        deferred_parcel_ids = set()
        for path in deferring_output_paths:
            expanded_paths += glob(path)
        for path in expanded_paths:
            with open(f'{path}/outputs.pkl', 'rb') as f:
                deferring_output = pickle.load(f)

            deferred_parcel_ids_train = set([cache_key['patch_id']
                                             for cache_key in deferring_output['density_estimate']['train_cache_misses']])
            deferred_parcel_ids_test = set([cache_key['patch_id']
                                             for cache_key in deferring_output['density_estimate']['test_cache_misses']])
            deferred_parcel_ids = deferred_parcel_ids.union(deferred_parcel_ids_train, deferred_parcel_ids_test)

        return dict(deferred_parcel_ids=deferred_parcel_ids)

    @staticmethod
    def patch_dataset(cfg, input_):
        print(f'Step: patch dataset ({cfg})')

        n_train_samples_prior = cfg['n_train_samples_prior']
        n_train_samples = cfg['n_train_samples']
        n_test_samples = cfg['n_test_samples']
        patch_max_n_pixels = cfg['patch_max_n_pixels']
        included_original_classes = cfg['included_original_classes']
        n_classes = len(included_original_classes)
        px_filter = _PX_FILTERS[ cfg['px_filter']] if 'px_filter' in cfg and cfg['px_filter'] in _PX_FILTERS else None
        included_bands = cfg['included_bands'] if 'included_bands' in cfg else None
        class_index = [label for ix, label in enumerate(config['class_index']) if ix in included_original_classes]
        random_seed = pxs.RANDOM_SEED_DEFAULT
        if 'generate_random_seed' in cfg:
            if isinstance(cfg['generate_random_seed'], bool):
                if cfg['generate_random_seed']:
                    random_seed = np.random.randint(int((np.iinfo(np.uint32).max + time.time()) % (2**32 - 1)))
                    print('Generated random seed.')
            elif isinstance(cfg['generate_random_seed'], int):
                random_seed = cfg['generate_random_seed']
                print('Using configured random seed.')

            print(f'Using random seed for parcel sampling: {random_seed}')

        if 'patch_intersection' in input_:
            train_eligible_parcel_ids, test_eligible_parcel_ids = set(ft.reduce(op.iconcat, input_['patch_intersection']['train'], [])), \
                                                                  set(ft.reduce(op.iconcat, input_['patch_intersection']['test'], []))
        else:
            train_eligible_parcel_ids, test_eligible_parcel_ids = None, None



        def _dataset_partition_stats(partition_name, partition):
            print(f'Dataset partition: {partition_name}')
            df = pd.DataFrame()
            df['class_ixs'] = included_original_classes
            df['class_name'] = class_index
            df['n_patches'] = [len(partition[class_ix].patch_data) for class_ix in range(n_classes)]


            df['n_pixels'] = [np.sum(list(patch.band_pixels.shape[0] for patch in partition[class_ix].patch_data))
                              for class_ix in range(n_classes)]

            return df

        def _eligible_patch_filter(eligible_parcel_ids):
            return lambda patch: patch.patch_properties['parcelID'] in eligible_parcel_ids

        train_path = cfg['train_path']
        test_path = cfg['test_path']
        print(f'Reading {train_path} ...')
        with open(train_path, 'rb') as f:
            class_train_datasets = pickle.load(f)
            patch_filter = _eligible_patch_filter(train_eligible_parcel_ids) if train_eligible_parcel_ids else None
            n_samples_to_take = n_train_samples_prior + n_train_samples if n_train_samples else None
            class_train_datasets = [ds.filter(bands=included_bands, patch_filter=patch_filter, px_filter=px_filter) \
                                      .sample(n_samples_to_take,
                                              patch_filter=_patch_px_limiter(patch_max_n_pixels),
                                              random_seed=random_seed)
                                    for ix, ds in enumerate(class_train_datasets)
                                    if ix in included_original_classes]

            #class_prior_patches = [datasets.patch_data[n_train_samples:] for datasets in class_train_datasets]
            for ds in class_train_datasets:
                if n_train_samples:
                    ds.patch_data = ds.patch_data[:n_train_samples]

        print(f'Reading {test_path} ...')
        with open(test_path, 'rb') as f:
            class_test_datasets = pickle.load(f)
            patch_filter = _eligible_patch_filter(test_eligible_parcel_ids) if test_eligible_parcel_ids else None
            class_test_datasets = [ds.filter(bands=included_bands, patch_filter=patch_filter, px_filter=px_filter) \
                                     .sample(n_test_samples, patch_filter=_patch_px_limiter(patch_max_n_pixels),
                                             random_seed=random_seed)
                                   for ix, ds in enumerate(class_test_datasets)
                                   if ix in included_original_classes]

        auxiliary_data_train = _extract_patch_ids(class_train_datasets), _extract_patch_stats(class_train_datasets)
        auxiliary_data_test = _extract_patch_ids(class_test_datasets), _extract_patch_stats(class_test_datasets)

        dataset_train_stats = _dataset_partition_stats('Train', class_train_datasets)
        dataset_test_stats = _dataset_partition_stats('Test', class_test_datasets)

        return dict(
            class_train_datasets=class_train_datasets,
            class_test_datasets=class_test_datasets,
            auxiliary_data_train=auxiliary_data_train,
            auxiliary_data_test=auxiliary_data_test,
            dataset_train_stats=dataset_train_stats,
            dataset_test_stats=dataset_test_stats,
            random_seed=random_seed
        )
