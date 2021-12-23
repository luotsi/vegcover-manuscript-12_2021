import argparse
import copy
import itertools
import os
import pickle
import shutil
import traceback
from datetime import datetime
from math import ceil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from classifiers import Classifiers
from config import config
from dataset_reader import DatasetReaders, PatchIntersection
from density_estimators import DensityEstimators
from pixel_sampler import PatchDataset, Patch

_USE_SAMPLE_CACHE = config['pipeline']['use_cached_densities']
_CACHE_INDEX_ROOT = config['pipeline']['cache_index_root']
_TIMEOUT_PARALLEL_S = 900


def map_parallel(parent_cfg, prev_outputs):
    data = []
    for batch_conf in parent_cfg['tasks']:
        data.append([batch_conf, prev_outputs['output_dir'], None, prev_outputs, True])

    with Pool(5) as p:
        jobs = [p.apply_async(run_batch, d) for d in data]
        results=[job.get(timeout=_TIMEOUT_PARALLEL_S) for job in jobs]
    export_output_keys = parent_cfg['outputs']
    exportable_outputs = {key: output_dict[key] for key in export_output_keys for output_dict in results if key in output_dict}
    return exportable_outputs


import multiprocessing as mp
def map_shards_parallel(parent_cfg, prev_outputs):
    n_workers = parent_cfg['n_workers']
    batch_conf_to_distribute = parent_cfg['task']
    distributable_data_keys = parent_cfg['distributable_data']
    output_dir = prev_outputs['output_dir']
    specific_data = prev_outputs
    for key in distributable_data_keys:
        specific_data = specific_data[key]
    batches = np.array_split(list(specific_data)[:n_workers], n_workers)
    distributable_data = []
    mp.set_start_method('fork')
    queue = mp.Queue()
    subprocesses = []
    for batch_ix, batch in enumerate(batches):
        prev_outputs_batch = {distributable_data_keys[0]:{distributable_data_keys[1]: batch.tolist()}}
        args = [batch_conf_to_distribute, output_dir,  None, prev_outputs_batch]
        worker_process = mp.Process(target=run_batch_fork, args=tuple([queue] + args))
        subprocesses.append(worker_process)
        worker_process.start()

    for worker_process in subprocesses:
        worker_process.join()

    results = []

    for ix in range(n_workers):
        results.append(queue.get())

    return {'sampled_density_file_paths': results}


STEP_HANDLERS = {
    'map_parallel': {
        'map_parallel': map_parallel,
        'map_shards_parallel': map_shards_parallel
    },
    'reduce_concatenate_densities': {
        'reduce_concatenate_densities': DensityEstimators.reduce_concatenate_densities
    },
    'patch_intersection': {
        'patch_intersection': PatchIntersection.intersection
    },
    'dataset_filter': {
        'patch_dataset': DatasetReaders.patch_dataset,
        'patch_dataset_deferred': DatasetReaders.patch_dataset_deferred
    },
    'density_estimate': {
        'gkde': DensityEstimators.gkde,
        'ndhistogram': DensityEstimators.ndhistogram,
        'lgpde': DensityEstimators.lgpde,
        'lgpde_deferred': DensityEstimators.lgpde_deferred,
        'lgpde_approx': DensityEstimators.lgpde_approx
    },
    'density_debug': {
        'plot_densities': DensityEstimators.plot_densities,
    },
    'classifier': {
        'nearest_neighbor_classifier': Classifiers.nearest_neighbor_classifier,
        'sklearn_classifier': Classifiers.sklearn_classifier,
        'sklearn_classifier_full_bayes': Classifiers.sklearn_classifier_full_bayes,
    }
}


def run_batch_wrapper(data):
    run_batch(*data)


def batch_config(batch_config_file_path: str):
    with open(batch_config_file_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _load_cached_output(path):
    print(f'Loading cached output from {path} ...')
    if path.endswith('.npy'):
        return np.load(path, allow_pickle=True)
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)


def _load_all_cache_indices():
    global_cache_index = {}
    print(f'Searching for cache indices below {_CACHE_INDEX_ROOT} ...')
    for path in Path(_CACHE_INDEX_ROOT).rglob('cache_index.pkl'):
        print(f'Adding to global cache index from {path.resolve()} ...')
        with open(str(path.resolve()), 'rb') as f:
            cache_index = pickle.load(f)
        for k, v in cache_index.items():
            global_cache_index[k] = v
    print(f'Enlisted {len(global_cache_index)} precomputed densities from {_CACHE_INDEX_ROOT}')
    return global_cache_index


def run_batch_fork(queue: mp.Queue, batch_conf, output_dir, variant_cache=None, prev_outputs=None, sub_batch=False):
    outputs = run_batch(batch_conf, output_dir, variant_cache, prev_outputs, sub_batch)
    queue.put(outputs)

def run_batch(batch_conf, output_dir, variant_cache=None, prev_outputs=None, sub_batch=False):
    outputs = prev_outputs if prev_outputs else {}
    outputs['output_dir'] = output_dir
    if not sub_batch and _USE_SAMPLE_CACHE:
        outputs['global_cache_index'] = _load_all_cache_indices()
        outputs['batch_cache_index'] = {}
    if not sub_batch and variant_cache and 'class_sample_densities_test' in variant_cache:
        print('Making cached test densities available for variant.')
        outputs['class_sample_densities_test'] = variant_cache['class_sample_densities_test']
    if not sub_batch and variant_cache and 'test_class_post_f_samples' in variant_cache:
        print('Making cached raw test density samples available for variant.')
        outputs['test_class_post_f_samples'] = variant_cache['test_class_post_f_samples']

    for step_conf in batch_conf['pipeline']:
        if step_conf['step'] in STEP_HANDLERS:
            if step_conf['name'] in STEP_HANDLERS[step_conf['step']]:

                step_handler = STEP_HANDLERS[step_conf['step']][step_conf['name']]

                if not ('skip' in step_conf and step_conf['skip']):
                    output_name = step_conf['output'] if 'output' in step_conf else step_conf['name']
                    if not sub_batch and 'read_cached' in step_conf and _USE_SAMPLE_CACHE:
                        cache_conf = step_conf['read_cached']
                        if isinstance(cache_conf, bool) and cache_conf:  # unfiltered outputs from previous run
                            results = _load_cached_output(f'{output_dir}/outputs.pkl')[output_name]
                        elif isinstance(cache_conf, str): # unfiltered outputs from previous given .out file
                            results = _load_cached_output(cache_conf)[output_name]
                        elif isinstance(cache_conf, dict): # each output entry from its own file (not .out)
                            results = {output_key: _load_cached_output(pickle_path)
                                       for (output_key, pickle_path) in cache_conf.items() if pickle_path}
                        elif isinstance(cache_conf, list): # filter outputs from previous .out file
                            results = {}
                            for conf in cache_conf:
                                output = _load_cached_output(conf['file'])[output_name]
                                output = {key: value for (key, value) in output.items() if key in conf['outputs']}
                                results = {**results, **output}
                        else:
                            results = None
                    else:
                        results = step_handler(step_conf, outputs)

                    outputs[output_name] = results

                    if not sub_batch and variant_cache is not None and 'class_sample_densities_test' in results:
                        print('Keeping test densities available in variant cache.')
                        variant_cache['class_sample_densities_test'] = results['class_sample_densities_test']

                    if not sub_batch and variant_cache is not None and 'test_class_post_f_samples' in results:
                        variant_cache['test_class_post_f_samples'] = results['test_class_post_f_samples']
            else:
                raise Exception(f"Unknown handler: {step_conf['step']}/{step_conf['name']}")
        else:
            raise Exception(f"Unknown step type in batch conf: {step_conf['step']}")

    if _USE_SAMPLE_CACHE:
        if len(outputs['batch_cache_index']) > 0:
            with open(f'{output_dir}/cache_index.pkl', 'wb') as f:
                pickle.dump(outputs['batch_cache_index'], f)

    return outputs


def _save_summary(batch_conf, outputs, output_dir):
    steps = batch_conf['pipeline']
    common_columns = ['density_estimator', 'experiment_name', 'sensor_type', 'dim', 'extreme_bins', 'classes',
                      'dest_params', 'n_train_samples', 'n_test_samples', 'patch_max_n_pixels', 'iteration_ix',
                      'random_seed']
    classifier_columns = ['classifier', 'preprocessing', 'classif_params', 'acc', 'class_acc']
    common_values = [steps[1]['name'],
                     batch_conf['experiment_name'],
                     steps[0]['data_source'],
                     steps[1]['n_bins_per_dim'],
                     str(steps[1]['extreme_bins'] if 'extreme_bins' in steps[1] else steps[1]['params']['extreme_bins']),
                     str(steps[0]['included_original_classes']) if 'included_original_classes' in steps[0] else "",
                     str(steps[1]['params']) if 'params' in steps[1] else "",
                     steps[0]['n_train_samples'],
                     steps[0]['n_test_samples'],
                     steps[0]['patch_max_n_pixels'] if 'patch_max_n_pixels' in steps[0] else "",
                     outputs['iteration_ix'],
                     outputs['patch_dataset']['random_seed'] if 'patch_dataset' in outputs else ""]

    rows = []
    for step in steps:
        if step['step'] == 'classifier' and ('skip' not in step or not step['skip']):
            classifier_acc = outputs[step['output']]['acc']
            classifier_prep = step['normalizer']
            classifier_params = str(step['params']) if 'params' in step else ""
            classifier_class_acc = str(np.diag(outputs[step['output']]['cm']).tolist())
            classifier_values = [step['output'], classifier_prep, classifier_params, classifier_acc, classifier_class_acc]
            values = common_values + classifier_values
            rows.append(values)
    df = pd.DataFrame(rows, columns=common_columns + classifier_columns)
    df.to_csv(f'{output_dir}/summary.csv', index=False, sep='|')
    df.to_pickle(f'{output_dir}/summary_df.pkl')


def _save_outputs(batch_conf, output_dir, outputs):
    outputs_to_save = {}
    for output_name in batch_conf['save_output_for_steps']:
        if isinstance(output_name, str):
            if output_name in outputs:
                outputs_to_save[output_name] = outputs[output_name]
        elif isinstance(output_name, dict):
            output_subset_name = list(output_name.keys())[0]
            outputs_to_save[output_subset_name] = \
                {property_name: outputs[output_subset_name][property_name]
                 for property_name in output_name[output_subset_name]}

    output_data_file_path = f'{output_dir}/outputs.pkl'
    with open(output_data_file_path, 'wb') as f:
        pickle.dump(outputs_to_save, f)


def _output_dir_path(batch_conf, batch_conf_file_path, variant_ix=None, iteration_ix=None):
    variant_suffix = "" if variant_ix is None else f'.{variant_ix}'
    iteration_suffix = "" if iteration_ix is None else f'.{iteration_ix}'
    timestamp_str = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    dirname_suffix = f'.{timestamp_str}.out' if batch_conf['output_timestamp_suffix'] else '.out'
    output_dir = batch_conf_file_path.replace('.yml', f'{variant_suffix}{iteration_suffix}{dirname_suffix}')
    return output_dir


def run_batches(batch_config_file_paths: [str]):
    for batch_config_file_path in batch_config_file_paths:
        with open(batch_config_file_path) as f:

            batch_conf = yaml.load(f, Loader=yaml.FullLoader)

            if 'experiment_name' not in batch_conf:
                experiment_name = os.path.basename(batch_config_file_path).replace('.yml', '')
                batch_conf['experiment_name'] = experiment_name

            output_dir = _output_dir_path(batch_conf, batch_config_file_path) if batch_conf['save_outputs'] else None
            try:
                outputs = run_batch(batch_conf, output_dir)
                if batch_conf['save_outputs']:
                    Path(output_dir).mkdir(exist_ok=True)
                    print(f'Outputs to save to {output_dir}:', *batch_conf['save_output_for_steps'], sep='\n\t- ')
                    _save_outputs(batch_conf, output_dir, outputs)
                    shutil.copy(batch_config_file_path, output_dir)
                    if (len(batch_conf['pipeline']) > 1 and
                            not('suppress_summary' in config['pipeline'] and
                                config['pipeline']['suppress_summary'])):
                        _save_summary(batch_conf, outputs, output_dir)
                    else:
                        print('Skipping output of summary statistics.')
                    os.system('git rev-parse HEAD > ' + output_dir + '/pipeline_version_git_hash')
            except:
                traceback.print_exc()


def run_variants(batch_config_file_path, batch_template, batch_conf_descrs, batch_pipeline_variants):
    for variant_ix, batch_pipeline_variant in enumerate(batch_pipeline_variants):
        if 'select_variants' in batch_template:
            if batch_template['select_variants'] is not None and variant_ix + 1 < batch_template['select_variants']:
                continue
            # currently only "from index" supported i.e. from given ix

        variant_descr_dict = batch_conf_descr_to_dict(batch_conf_descrs[variant_ix])
        print(f'Processing variant {variant_ix + 1}/{len(batch_pipeline_variants)}: {variant_descr_dict}...')

        batch_conf = copy.deepcopy(batch_template)
        batch_conf['pipeline'] = batch_pipeline_variant

        n_iterations = batch_conf['n_iterations'] if 'n_iterations' in batch_conf else 1
        variant_cache = dict()
        for iteration_ix in range(n_iterations):
            if n_iterations > 1:
                print(f'Iterating variant; iteration {iteration_ix + 1} out of {n_iterations}')


            output_dir = _output_dir_path(batch_template, batch_config_file_path, variant_ix=variant_ix, iteration_ix=iteration_ix) \
                if batch_template['save_outputs'] else None
            load_cache_from_dir = output_dir
            try:
                outputs = run_batch(batch_conf, load_cache_from_dir, variant_cache=variant_cache)
                outputs['iteration_ix'] = iteration_ix

                if batch_template['save_outputs']:
                    _output_and_summarize(variant_descr_dict, batch_conf, batch_config_file_path, output_dir, outputs)

            except:
                traceback.print_exc()


def _output_and_summarize(variant_batch_conf_descr, batch_conf, batch_config_file_path, output_dir, outputs):
    Path(output_dir).mkdir(exist_ok=True)
    if variant_batch_conf_descr:
        with open(f'{output_dir}/variant_descr.yml', 'w') as f:
            yaml.dump(variant_batch_conf_descr, f)

    print(f'Outputs to save to {output_dir}:', *batch_conf['save_output_for_steps'], sep='\n\t- ')
    _save_outputs(batch_conf, output_dir, outputs)
    shutil.copy(batch_config_file_path, output_dir)
    _save_summary(batch_conf, outputs, output_dir)
    os.system('git rev-parse HEAD > ' + output_dir + '/pipeline_version_git_hash')


def conf_variant(conf_template, variantspec):
    new_conf = copy.deepcopy(conf_template)
    step_ix = {step['id']: ix for ix, step in enumerate(conf_template)}
    for mutation in variantspec:
        key_path, val = mutation
        step = new_conf[step_ix[key_path[0]]]
        conf = step
        key_path = key_path[1:]
        for key in key_path[:-1]:
            conf = conf[key]
        conf[key_path[-1]] = val
    return new_conf


def batch_conf_descr_to_dict(batch_conf_descr):
    root = {}
    for kv in batch_conf_descr:
        key_path, value = kv
        cur = root
        for ix, key in enumerate(key_path):
            if ix < len(key_path) - 1:
                if key not in cur:
                    cur[key] = {}
                cur = cur[key]
            else:
                cur[key] = value
    return root


def batch_config_variations(batch_template_file_path):
    with open(batch_template_file_path) as f:
        conf_template = yaml.load(f, Loader=yaml.FullLoader)
    variations = conf_template['variations']
    if 'experiment_name' not in conf_template:
        experiment_name = os.path.basename(batch_template_file_path).replace('.yml', '')
        conf_template['experiment_name'] = experiment_name
    path_value_list = []
    for variant in variations:
        variant_as_list = []
        for field, values in variant.items():
            if field == 'id':
                continue
            for value in values:
                variant_as_list.append(([variant['id'], field], value))
        path_value_list.append(variant_as_list)
    batch_conf_descrs = list(itertools.product(*path_value_list))
    return conf_template, batch_conf_descrs, [conf_variant(conf_template['pipeline'], batch_conf_descr)
                                              for batch_conf_descr in batch_conf_descrs]


def main(args):
    if args.batch_template_file_path:
        conf_template, batch_conf_descrs, batch_confs = batch_config_variations(args.batch_template_file_path)
        # pprint.PrettyPrinter(indent=2).pprint(batch_conf_descr_to_dict(batch_conf_descrs))
        run_variants(args.batch_template_file_path, conf_template, batch_conf_descrs, batch_confs)
    elif args.batch_config_files:
        run_batches(args.batch_config_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--batch_template_file_path',
                        type=str,
                        help='Template pipeline config with variations spec')

    parser.add_argument('batch_config_files', nargs=argparse.REMAINDER)

    args = parser.parse_args()
    main(args)
