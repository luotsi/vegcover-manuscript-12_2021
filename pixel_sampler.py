"""
TODO help text
"""

from __future__ import annotations

import argparse
import pickle
import textwrap
from dataclasses import dataclass
from functools import reduce
from typing import Dict
import hashlib
import os

import fiona
import numpy as np
import rasterio
import rasterio.coords
import rasterio.mask
import rasterio.sample
import shapely
import shapely.geometry
from tqdm import tqdm

from config import config

MIN_NUM_PIXELS_PER_PATCH = 5
RANDOM_SEED_DEFAULT = 789

DEFAULT_MSI_BAND_INDEX = config['pixel_sampler']['raster_files']['s2_ndti_ndvi']['band_index']
HASH_SALT = os.environ.get('HASH_SALT', default=str(np.random.randint(np.iinfo('int32').max)))
print(f'Generated new random hash salt: {HASH_SALT} .')

def digest(string):
    h = hashlib.sha224(string.encode('utf-8'))
    h.update(HASH_SALT.encode('utf-8'))
    return h.hexdigest()

def reverse_band_index(band_index):
    return {band_name: ix for ix, band_name in enumerate(band_index)}


def msi_patch_stats(patch:Patch, band_index=DEFAULT_MSI_BAND_INDEX):
    rev_band_index = reverse_band_index(band_index)
    return [patch.band_pixels.shape[0],
            np.mean(patch.band_pixels[:, rev_band_index['p_cld']]),
            np.mean(patch.band_pixels[:, rev_band_index['p_snw']])]


def sar_patch_stats(patch:Patch):
    return [patch.band_pixels.shape[0]]


def include_parcel_ids_filter(eligible_parcel_ids):
    def filter_(patch: Patch):
        return patch.patch_properties['parcelID'] in eligible_parcel_ids
    return filter_


F_PATCH_STATS = {
    'msi_patch_stats': msi_patch_stats,
    'sar_patch_stats': sar_patch_stats
}


class Patch:

    def __init__(self, band_pixels: [np.ndarray], patch_properties: Dict[int, str], patch_stats=None, f_patch_stats=None):
        self.band_pixels = band_pixels
        self.patch_properties = patch_properties

        self.patch_stats = patch_stats

        if f_patch_stats:
            self.patch_stats = f_patch_stats(self)

    def filter(self, bands: [int] = None, px_filter=None, filter_out_nans=True) -> Patch:
        patch = self

        if filter_out_nans:
            finite_ix = np.all(np.isfinite(patch.band_pixels), axis=1)
            if np.sum(finite_ix) < patch.band_pixels.shape[0]:
                #print(f'Filtering out {patch.band_pixels.shape[0] - np.sum(finite_ix)} NaN pixels from parcel {patch.patch_properties["parcelID"]} (had {patch.band_pixels.shape[0]} px)')
                patch.band_pixels = patch.band_pixels[finite_ix, :]

        if px_filter:
            patch = Patch(px_filter(patch.band_pixels), patch.patch_properties, patch_stats=patch.patch_stats)

        if bands:
            patch = Patch(patch.band_pixels[:, bands], patch.patch_properties, patch_stats=patch.patch_stats)

        return patch


def _patch_properties_filter(properties):
    return lambda patch: \
        reduce((lambda acc, e: acc and patch.patch_properties[e[0]] == e[1]), properties.items(), True)


def filter_dataset_patches(dataset, patch_filter):
    return PatchDataset(dataset.raster_file_name, dataset.patch_shapefile_name, dataset.band_ix,
                        [patch for patch in dataset.patch_data
                         if patch_filter(patch)])

class PatchDataset:
    def __init__(self, raster_file_name: str, patch_shapefile_name: str, band_ix: [str], patches=None, f_patch_stats=None):

        def bb_to_polygon(bb: rasterio.coords.BoundingBox) -> shapely.geometry.Polygon:
            return shapely.geometry.Polygon([(bb.left, bb.top),
                                             (bb.right, bb.top),
                                             (bb.right, bb.bottom),
                                             (bb.left, bb.bottom),
                                             (bb.left, bb.top)])

        if patches is None:
            patches = []
            print(f'Opening raster {raster_file_name} for reading...')
            with rasterio.open(raster_file_name) as raster_dataset:
                bounding_box = bb_to_polygon(raster_dataset.bounds)
                print(f'Opening shapefile {patch_shapefile_name} for reading...')
                with fiona.open(patch_shapefile_name, 'r') as shapefile:
                    print(f'Filtering AOIs within raster bounds from {len(shapefile)} polygons, extracting pixels...')
                    for feature in tqdm(shapefile):
                        shape = shapely.geometry.shape(feature['geometry'])
                        if bounding_box.contains(shape):
                            patch_data, _ = rasterio.mask.mask(raster_dataset, [shape], crop=True, filled=False)
                            band_pixels = np.array([patch_data[i].compressed() for i in range(patch_data.shape[0])]).T
                            if config['pipeline']['do_anonymize'] and feature['properties']['parcelID'] is not None:
                                feature['properties']['parcelID'] = digest(feature['properties']['parcelID'])
                            patch_properties = feature['properties']
                            if band_pixels.shape[0] >= MIN_NUM_PIXELS_PER_PATCH:
                                patch = Patch(band_pixels, patch_properties, f_patch_stats=f_patch_stats)
                                patches.append(patch)
            print(f'Extracted {len(patches)} patches from AOI.')
        self.raster_file_name: str = raster_file_name
        self.patch_shapefile_name: str = patch_shapefile_name
        self.band_ix: [str] = band_ix
        self.patch_data: [Patch] = patches

    @classmethod
    def from_patch_data(cls, dataset: PatchDataset, patch_data):
        return cls(dataset.raster_file_name, dataset.patch_shapefile_name, dataset.band_ix, patch_data)


    def merge_single(self, dataset):
        self.patch_data += dataset.patch_data
        self.patch_shapefile_name += '|' + dataset.patch_shapefile_name


    @staticmethod
    def merge(class_datasets, merge_specs: [dict]):
        merged_ix = []
        for merge_spec in merge_specs:
            merge_into_ix = merge_spec['merge_into']
            for merge_from_ix in merge_spec['merge_what']:
                class_datasets[merge_into_ix].patch_data += class_datasets[merge_from_ix].patch_data
                class_datasets[merge_into_ix].patch_shapefile_name += '|' + class_datasets[merge_from_ix].patch_shapefile_name
                merged_ix.append(merge_from_ix)
        for ix_merged in reversed(merged_ix):
            del class_datasets[ix_merged]

        return class_datasets


    @staticmethod
    def load(file_name) -> [PatchDataset]:
        with open(file_name, 'rb') as f:
            return pickle.load(f)

    def save(self, file_name: str):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    def sample(self, n, random_seed=RANDOM_SEED_DEFAULT, patch_filter=lambda x: True):
        if random_seed:
            np.random.seed(random_seed)

        patch_data = [patch for patch in self.patch_data if patch_filter(patch)]

        if n is None:
            index = range(len(patch_data))
        else:
            if n > len(patch_data):
                raise Exception(f'Cannot sample {n} when population is {len(patch_data)}')
            index = np.random.choice(len(patch_data), len(patch_data), replace=False)[:n]

        return self.from_patch_data(self, [patch_data[ix] for ix in index])

    def filter(self, properties: dict = None, bands: [int] = None, patch_filter=None, px_filter=None) -> PatchDataset:
        """
        Return a new patch dataset as a subset of self, based on the given required patch properties.
        :param properties:
        :param bands
        :return:
        """

        dataset = self

        # print(f'Patch count before any filtering: {len(dataset.patch_data)}...')
        if bands or px_filter:
            if bands:
                filtered_band_ix = [dataset.band_ix[band] for band in bands]
            else:
                filtered_band_ix = list(range(dataset.patch_data[0].band_pixels.shape[1]))
            patches = [patch.filter(bands=bands, px_filter=px_filter) for patch in dataset.patch_data]
            dataset = PatchDataset(dataset.raster_file_name, dataset.patch_shapefile_name, filtered_band_ix,
                                   [patch for patch in patches if patch.band_pixels.shape[0] > MIN_NUM_PIXELS_PER_PATCH])

        if properties:
            print(f'Filtering w/given properties filter with {len(dataset.patch_data)} patches initially...')
            dataset = filter_dataset_patches(dataset, _patch_properties_filter(properties))
            print(f'Remaining patch count after properties filtering: {len(dataset.patch_data)}...')

        if patch_filter:
            print(f'Filtering w/given patch filter with {len(dataset.patch_data)} patches initially...')
            dataset = filter_dataset_patches(dataset, patch_filter)
            print(f'Remaining patch count after patch filtering: {len(dataset.patch_data)}...')


        return dataset


def generate_datasets(raster_path, shapefile_paths, bands, band_index, output_dir, raster_file_label):
    class_train_datasets = [(PatchDataset(raster_path, shapefile_path, band_index, f_patch_stats=sar_patch_stats)
                             .filter(properties={'split': 'train'}, bands=bands)) for shapefile_path in shapefile_paths]

    class_test_datasets = [(PatchDataset(raster_path, shapefile_path, band_index, f_patch_stats=sar_patch_stats)
                            .filter(properties={'split': 'test'}, bands=bands)) for shapefile_path in shapefile_paths]
    if 'merge_classes' in config['pixel_sampler']:
        class_train_datasets = PatchDataset.merge(class_train_datasets, config['pixel_sampler']['merge_classes'])
        class_test_datasets = PatchDataset.merge(class_test_datasets, config['pixel_sampler']['merge_classes'])

    train_output__file_path = f'{output_dir}/train_{raster_file_label}.pkl'
    test_output__file_path = f'{output_dir}/test_{raster_file_label}.pkl'

    print(f'Writing {train_output__file_path} ...')
    with open(train_output__file_path, 'wb') as f:
        pickle.dump(class_train_datasets, f)

    print(f'Writing {test_output__file_path} ...')
    with open(test_output__file_path, 'wb') as f:
        pickle.dump(class_test_datasets, f)


def main(args):
    f_patch_stats = None if not args.f_patch_stats else F_PATCH_STATS[args.f_patch_stats]

    if args.generate_from_conf:
        output_dir = config['pixel_sampler']['output_dir']
        shapefile_root = config['pixel_sampler']['shapefile_root']
        shapefile_names = config['pixel_sampler']['class_shapefiles']
        shapefile_paths = [f'{shapefile_root}/{shapefile_name}' for shapefile_name in shapefile_names]
        for raster_file_label in config['pixel_sampler']['raster_files'].keys():
            raster_path = config['pixel_sampler']['raster_files'][raster_file_label]['file_name']
            band_index = config['pixel_sampler']['raster_files'][raster_file_label]['band_index']
            bands = [band for band in range(len(band_index))]
            generate_datasets(raster_path, shapefile_paths, bands, band_index, output_dir, raster_file_label)
    else:
        dataset = PatchDataset(args.in_source_tif, args.in_aoi_shapefile, config['pixel_sampler']['band_index'], f_patch_stats=f_patch_stats)
        output_path = f'{args.output_dir}/{args.output_file_name}'
        print(f'Saving dataset in {output_path} ...')
        dataset.save(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--in_source_tif',
                        type=str,
                        help='Geotiff/bigtiff geospatial image (.tif)')
    parser.add_argument('-a', '--in_aoi_shapefile',
                        type=str,
                        help='ESRI shapefile containing a set of polygons (.shp with its auxiliary files)')
    parser.add_argument('-d', '--output_dir',
                        help='Directory for output pickle file',
                        type=str,
                        default='.')

    parser.add_argument('-o', '--output_file_name',
                        help='File name for output pickle file',
                        type=str,
                        default='test.pkl')

    parser.add_argument('-f', '--f_patch_stats',
                        help='function name for extracting patch statistics.',
                        type=str)

    parser.add_argument('-g', '--generate_from_conf',
                        dest='generate_from_conf',
                        action='store_true',
                        help='Generate and merge datasets from config')

    args = parser.parse_args()
    main(args)
