import bz2
import copy
import os
import pickle
import traceback
from datetime import datetime
from io import StringIO
from pathlib import Path
from matplotlib import pyplot as plt
from math import ceil, sqrt

import matlab.engine
import numpy as np
import pystan
from scipy.stats import gaussian_kde
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import normalize
from tqdm import tqdm
import pixel_sampler

from config import config
from histogram import PatchDatasetBandHistograms
from redirect import suppress_stdout_stderr

_HYPERPRIOR_WEIGHT_ALPHA = 1
_LGPDE_CACHE_KEYS = ['step_name', 'patch_id', 'n_pixels', 'n_bins_per_dim', 'extreme_bins', 'params']
_DO_COMPILE_STAN_MODELS = config['pipeline']['do_compile_stan_models']


def estimate_patch_densities(class_data, density_estimator, X_s, dim):
    print('Estimating densities.')

    class_densities = []
    class_stats = []
    output_log = {'stdout': '', 'stderr': ''}
    for class_ix in range(len(class_data)):
        patches = class_data[class_ix].patch_data
        print(f'Found {len(patches)} patches in class {class_ix}. Estimating patch densities...')
        class_patch_densities = []
        class_patch_stats = []
        for patch_ix, patch in tqdm(enumerate(patches)):
            patch_data = patch.band_pixels
            patch_stats = {'parcelID': patch.patch_properties['parcelID'],
                           'n_pixels': patch.band_pixels.shape[0]}

            try:
                density, stdout, stderr = density_estimator(X_s, patch_data, dim)
                if output_log:
                    output_log['stdout'] += f'\n: class {class_ix} patch {patch_ix}\n{stdout}'
                    output_log['stderr'] += f'\n: class {class_ix} patch {patch_ix}\n{stderr}'
            except IOError:
                traceback.print_exc()
                density = np.zeros(dim).ravel()
                density[:] = np.NaN
            except:
                traceback.print_exc()
                density = np.zeros(dim).ravel()
                density[:] = np.NaN
            class_patch_densities.append(density.ravel())
            class_patch_stats.append(patch_stats)

        class_densities.append(np.array(class_patch_densities))
        class_stats.append(class_patch_stats)
    return class_densities, class_stats, output_log


def density_estimate_gkde(X_s, X_data, dim):
    kde_estimator = gaussian_kde(X_data.T)
    return kde_estimator(X_s.T).reshape(dim).T, "", ""


def matlab_engine(matlab_gpstuff_dir):
    curdir = os.getcwd()
    os.chdir(matlab_gpstuff_dir)
    eng = matlab.engine.start_matlab()
    os.chdir(curdir)
    return eng


def density_estimate_lgpdens(eng, extreme_bins, basis):
    def f(X_s, X_data,dim):
        x = matlab.double(X_data.tolist())
        bin_bounds = matlab.double(extreme_bins)
        stdout, stderr = StringIO(), StringIO()
        assert dim[0] == dim[1]

        p,pt,xt=eng.lgpdens(x,
                            'range',
                            bin_bounds,
                            'gridn', dim[0],
                            'basis', basis,
                            nargout=3,
                            stdout=stdout,
                            stderr=stderr)
        p,pt,xt = np.asarray(p), np.asarray(pt), np.asarray(xt)
        return p.reshape(dim), stdout.getvalue(), stderr.getvalue()

    return f


def posterior_patch_samples_as_array(post_f_samples):
    n_classes, n_samples_per_class = len(post_f_samples), len(post_f_samples[0])
    all_class_patch_samples = []
    for class_ix in range(n_classes):
        class_patch_samples = []
        for m in range(n_samples_per_class):
            logp_post = post_f_samples[class_ix][m]['lp__'].reshape(-1,1)
            samples = post_f_samples[class_ix][m]['exp_f']
            patch_samples = np.hstack([logp_post, samples])
            class_patch_samples.append(patch_samples)
        all_class_patch_samples.append(class_patch_samples)

    return np.array(all_class_patch_samples)


def subset_posterior_patch_samples(samples_arr, n_classes, q=0.5, n=500):
    class_data = []
    for class_ix in range(n_classes):
        data = samples_arr[class_ix,:,:,1:]
        quantile_limit = np.quantile(samples_arr[class_ix,:,:,0], 1-q, axis=1)
        ix = np.moveaxis(np.moveaxis(samples_arr[class_ix, :, :, 0], 1,0) > quantile_limit, 1, 0)
        data = data[ix]
        downsample_ix = np.random.choice(data.shape[0], n, replace=True)
        class_data.append(data[downsample_ix])
    y = np.array(list([class_ix for class_ix in range(n_classes) for m in range(n)]))
    return class_data, y


def gp_kernel_matern(X1, X2, l=1, nu=.1):
    return Matern(length_scale=l, nu=nu)(X1,X2)


def gp_kernel_rbf(X1, X2, l=1, sigma_f=.1):
    '''
    Isotropic squared exponential kernel. Computes
    a covariance matrix from points in X1 and X2.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        Covariance matrix (m x n).
    '''
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


def class_pixels(class_train_datasets, offsets_vh_vv, class_index):
    pixels = np.vstack([class_train_datasets[class_index].patch_data[i].band_pixels
                        for i in range(len(class_train_datasets[class_index].patch_data))])
    pixels = pixels + offsets_vh_vv
    return pixels


def sample_lgp_posterior(y, M, n_pixels, Sigma, model, n_iter, n_chains, mu=None, print_summary=False):
    print(f'Sampling posterior from {n_pixels} pixels for {M} discretized variables')
    if mu is None:
        mu = np.zeros(M)
    #print(f'mu = {mu}')
    print(datetime.now())
    data = dict(y=y,
                m=M,
                n=n_pixels,
                mu=mu,
                alpha=_HYPERPRIOR_WEIGHT_ALPHA,
                Sigma=Sigma) #

    fit = model.sampling(data=data,
                         iter=n_iter,
                         chains=n_chains,
                         control=dict(adapt_delta=2, max_treedepth=12),
                         pars=['exp_f'])
    extracts = fit.extract(permuted=True)


    print(datetime.now())
    return extracts, str(fit)


def extract_pmu_stan(extracts, m=None):
    """
    Posterior sample mean
    """
    m = np.sqrt(extracts['exp_f'].shape[1]).astype(int)
    f_pmu = np.mean(extracts['exp_f'], axis=0).reshape(m, m)
    return f_pmu


def extract_map_stan(extracts, m=None):
    map_ix = np.argmax(extracts['lp__'])
    f_map = extracts['exp_f'][map_ix]
    if not m:
        m = np.sqrt(f_map.shape[0]).astype(int)
    f_map = f_map.reshape(m, m)
    return f_map


def h_2d_basis(X):
    x1, x2 = X[:, 0].reshape(-1,1), X[:, 1].reshape(-1,1) # x1, x2 : n x 1
    return np.hstack([x1, x1 ** 2, x2, x2 ** 2, x1 * x2]).T # , x1 ** 3, x2 ** 3


def h_ident_basis(X):
    return X.T


LGP_BASIS = {
    'h_2d_basis': (h_2d_basis, 5),
    'h_ident_basis': (h_ident_basis, 2)
}


def lgp_cov_prior(h_basis, dim_basis, X_s, l=1, beta=100):
    B = np.eye(dim_basis) * beta
    return gp_kernel_rbf(X_s, X_s, l=l) + h_basis(X_s).T @ B @ h_basis(X_s)


def _make_sample_dump_path(raw_samples_dump_dir, posterior_samples_with_metadata):
    Path(raw_samples_dump_dir).mkdir(parents=True, exist_ok=True)
    d = posterior_samples_with_metadata

    return f"{raw_samples_dump_dir}/lgpde_psamples__{d['partition_descr']}__b{d['n_bins_per_dim']}__vhvv_{':'.join([str(b) for b in d['extreme_bins']])}__" \
           f"{d['params']['basis']}__l{d['params']['length_scale']}__id{d['patch_id']}.pbz2"


def plot_2d_density(X_s, density, ax=None, title=None, xtitle=None, ytitle=None, plot_X=None, markersize=1, addrandom=1):
    if not ax:
        fig, ax = plt.subplots()

    density = density.T
    xmin, ymin = np.min(X_s, axis=0)
    xmax, ymax = np.max(X_s, axis=0)
    bounds = [xmin, xmax, ymin, ymax]
    ax.imshow(np.rot90(density), cmap=plt.cm.gist_earth_r, extent=bounds)
    if plot_X is not None:
        plot_X = plot_X + addrandom * (np.random.rand(*plot_X.shape) - .5)
        ax.plot(plot_X[:,1], plot_X[:,0], 'k.', markersize=markersize*(xmax-ymin)/20, color='r')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_title(title)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)

class DensityEstimators():

    #    return gp_kernel_matern(X_s, X_s, l=l) + h_basis(X_s).T @ B @ h_basis(X_s)

    _stan_model = None


    @classmethod
    def _get_stan_model(cls):
        if cls._stan_model is None and _DO_COMPILE_STAN_MODELS:
            cls._stan_model = pystan.StanModel(file="model_cholesky.stan")
        return cls._stan_model

    @staticmethod
    def reduce_concatenate_densities(cfg, prev_outputs):
        """
            - density_estimate_sar
            - density_estimate_msi
        """
        sar_class_sample_densities_train = prev_outputs['map_parallel']['density_estimate_sar']['class_sample_densities_train']
        msi_class_sample_densities_train = prev_outputs['map_parallel']['density_estimate_msi']['class_sample_densities_train']
        class_sample_densities_train = zip(sar_class_sample_densities_train, msi_class_sample_densities_train)
        concatenated_train_features = [np.hstack(class_densities) for class_densities in class_sample_densities_train]

        if 'class_sample_densities_test' in prev_outputs:
            concatenated_test_features = prev_outputs['class_sample_densities_test'] # recycle, it's an iteration of a variant if given
        else:
            sar_class_sample_densities_test = prev_outputs['map_parallel']['density_estimate_sar']['class_sample_densities_test']
            msi_class_sample_densities_test = prev_outputs['map_parallel']['density_estimate_msi']['class_sample_densities_test']
            class_sample_densities_test = zip(sar_class_sample_densities_test, msi_class_sample_densities_test)
            concatenated_test_features = [np.hstack(class_densities) for class_densities in class_sample_densities_test]
        return dict(class_sample_densities_train=concatenated_train_features, class_sample_densities_test=concatenated_test_features)

    @staticmethod
    def plot_densities(cfg, prev_outputs):
        class_sample_densities_train = prev_outputs['density_estimate']['class_sample_densities_train']
        #class_sample_densities_test = prev_outputs['density_estimate']['class_sample_densities_test']
        auxiliary_data_train = prev_outputs['patch_dataset']['auxiliary_data_train']
        #auxiliary_data_test = prev_outputs['patch_dataset']['auxiliary_data_test']
        de_method = 'LGPDE'
        n_classes = len(class_sample_densities_train)
        n_bins_per_dim = cfg['n_bins_per_dim']
        extreme_bins = cfg['extreme_bins']
        extreme_bins = (extreme_bins[0], extreme_bins[1]), (extreme_bins[2], extreme_bins[3])
        step = ((extreme_bins[0][1] - extreme_bins[0][0]) / n_bins_per_dim,
                (extreme_bins[1][1] - extreme_bins[1][0]) / n_bins_per_dim)

        vh_edges = list(np.arange(extreme_bins[0][0], extreme_bins[0][1] + step[0], step[0]))
        vv_edges = list(np.arange(extreme_bins[1][0], extreme_bins[1][1] + step[1], step[1]))
        vh, vv = np.meshgrid(vh_edges[:-1], vv_edges[:-1])
        X_s = np.vstack([vh.ravel(), vv.ravel()]).T
        dim = np.array([len(vh_edges) - 1, len(vv_edges) - 1])
        class_index = [label for ix, label in enumerate(config['class_index']) if ix in cfg['included_original_classes']]

        for class_ix, class_train_densities in enumerate(class_sample_densities_train):
            n_plot_rows = n_plot_cols = ceil(sqrt(len(class_train_densities)))

            fig, axs = plt.subplots(n_plot_rows, n_plot_cols, figsize=(20, 20), frameon=True, edgecolor='black')

            fig.tight_layout(rect=[0, 0.03, 1, 0.7], h_pad=7, w_pad=2)
            for sample_ix, density in enumerate(class_train_densities):
                parcel_id = auxiliary_data_train[0][class_ix][sample_ix]
                n_px = auxiliary_data_train[1][class_ix][sample_ix][0]
                row = int(sample_ix / n_plot_rows)
                col = sample_ix % n_plot_rows

                plot_2d_density(X_s, density.reshape(dim), ax=axs[row,col], title=f"{class_index[class_ix]}\nParcel {parcel_id}: {n_px}px",
                                xtitle='log(VV) / dB',
                                ytitle='log(VH) / dB')
            #plt.subplots_adjust(bottom=0.1, top=0.7)
            fig.suptitle(f'Density estimates for crop field texture classes using {de_method}\n on dual-pol SAR intensities (Sentinel-1)', weight='bold')
            output_dir = prev_outputs['output_dir']
            Path(output_dir).mkdir(exist_ok=True)

            plt.savefig(f"{output_dir}/density_debug_class_{class_ix}_{len(class_train_densities)}prcl_{dim}.png")
            plt.show()

    @staticmethod
    def gkde(cfg, prev_outputs):
        print(f'Step: GKDE ({cfg})')
        max_n_pixels = cfg['params']['max_n_pixels'] if 'max_n_pixels' in cfg['params'] else None
        n_bins_per_dim = cfg['n_bins_per_dim']
        extreme_bins = cfg['extreme_bins']
        class_train_datasets = prev_outputs['patch_dataset']['class_train_datasets']
        class_test_datasets = prev_outputs['patch_dataset']['class_test_datasets']

        print(f'Train set of [{cfg["id"]}] has {[len(ds.patch_data) for ds in class_train_datasets]} patches in each class.')
        print(f'Test set of [{cfg["id"]}] has {[len(ds.patch_data) for ds in class_test_datasets]} patches in each class.')

        extreme_bins = (extreme_bins[0], extreme_bins[1]), (extreme_bins[2], extreme_bins[3])
        step = ((extreme_bins[0][1] - extreme_bins[0][0]) / n_bins_per_dim,
                (extreme_bins[1][1] - extreme_bins[1][0]) / n_bins_per_dim)

        vh_edges = list(np.arange(extreme_bins[0][0], extreme_bins[0][1] + step[0], step[0]))
        vv_edges = list(np.arange(extreme_bins[1][0], extreme_bins[1][1] + step[1], step[1]))
        edges = np.array([vh_edges, vv_edges])
        dim = np.array([len(vh_edges) - 1, len(vv_edges) - 1])
        X_s_meshgrid = vh_test, vv_test = np.meshgrid(vh_edges[:-1], vv_edges[:-1])
        X_s = np.vstack([vh_test.ravel(), vv_test.ravel()]).T
        density_estimator = density_estimate_gkde

        ret = dict(
            X_s_meshgrid=X_s_meshgrid,
            X_s=X_s,
            edges=edges,
            vh_edges=vh_edges,
            vv_edges=vv_edges
        )

        if cfg['estimate_class_densities']:
            class_densities = []
            print('Estimating train set class-wise densities w/ GKDE...')
            for class_ix in range(len(class_train_datasets)):
                X_train = np.vstack([class_train_datasets[class_ix].patch_data[i].band_pixels
                                     for i in range(len(class_train_datasets[class_ix].patch_data))])
                n = X_train.shape[0]

                if max_n_pixels:
                    n_sample = min(max_n_pixels, n)
                    train_ix = np.random.choice(range(n), size=n_sample, replace=False)
                    X_train = X_train[train_ix, :]

                density, _, _ = density_estimator(X_s, X_train, dim)
                class_densities.append(density)
            ret['class_densities'] = class_densities

        print('Estimating train set per-sample densities w/ GKDE...')
        class_sample_densities_train, train_patch_stats, _ = estimate_patch_densities(class_train_datasets, density_estimator, X_s, dim)

        if 'class_sample_densities_test' in prev_outputs:
            print('Using cached class_sample_densities_test .')
            class_sample_densities_test, test_patch_stats = prev_outputs['class_sample_densities_test'], None
        else:
            print('Estimating test set per-sample densities w/ GKDE...')
            class_sample_densities_test, test_patch_stats, _ = estimate_patch_densities(class_test_datasets,
                                                                                        density_estimator,
                                                                                        X_s,
                                                                                        dim)

        ret['class_sample_densities_train'] = class_sample_densities_train
        ret['class_sample_densities_test'] = class_sample_densities_test
        ret['train_patch_stats'] = train_patch_stats
        ret['test_patch_stats'] = test_patch_stats
        return ret

    @staticmethod
    def lgpde_approx(cfg, prev_outputs):
        print(f'Step: LGPDE (Gpstuff approximation) ({cfg})')
        dim = [cfg['n_bins_per_dim'], cfg['n_bins_per_dim']]
        extreme_bins = cfg['extreme_bins']
        X_s = None # Not needed for the gpstuff implementation
        basis = cfg['params']['basis']
        matlab_gpstuff_dir = cfg['params']['matlab_gpstuff_dir']
        max_n_pixels = cfg['params']['max_n_pixels'] if 'max_n_pixels' in cfg['params'] else None
        class_train_datasets = prev_outputs['patch_dataset']['class_train_datasets']
        class_test_datasets = prev_outputs['patch_dataset']['class_test_datasets']
        eng = matlab_engine(matlab_gpstuff_dir)
        density_estimator = density_estimate_lgpdens(eng, extreme_bins, basis)


        try:

            ret = {}
            if cfg['estimate_class_densities']:
                print('Estimating train set class-wise densities w/ LGPDE (GPStuff/Laplace)...')
                class_densities = []
                output_log = {'stdout': "", 'stderr': ""}
                for class_ix in range(len(class_train_datasets)):
                    X_train = np.vstack([class_train_datasets[class_ix].patch_data[i].band_pixels
                                         for i in range(len(class_train_datasets[class_ix].patch_data))])

                    n = X_train.shape[0]
                    if max_n_pixels:
                        n_sample = min(max_n_pixels, n)
                        train_ix = np.random.choice(range(n), size=n_sample, replace=False)
                        X_train = X_train[train_ix, :]
                    else:
                        n_sample = n

                    print(f'Estimating density of class {class_ix} with {n_sample} data points...')
                    density, stdout, stderr = density_estimator(X_s, X_train, dim)
                    class_densities.append(density)
                    if output_log:
                        output_log['stdout'] += f'\n: Class {class_ix}: \n{stdout}'
                        output_log['stderr'] += f'\n: Class {class_ix}: \n{stderr}'
                ret['class_densities'] = class_densities
                ret['class_train_log'] = output_log

            print('Estimating train set per-sample densities w/ LGPDE (Gpstuff)...')
            class_sample_densities_train, train_patch_stats, train_output_log = estimate_patch_densities(class_train_datasets, density_estimator, X_s, dim)

            if 'class_sample_densities_test' in prev_outputs:
                print('Using cached class_sample_densities_test .')
                class_sample_densities_test, test_patch_stats = prev_outputs['class_sample_densities_test'], None
                test_output_log = "N/A (using cached densities for test set)"
            else:
                print('Estimating test set per-sample densities w/ LGPDE (Gpstuff)...')
                class_sample_densities_test, test_patch_stats, test_output_log = estimate_patch_densities(class_test_datasets,
                                                                       density_estimator,
                                                                       X_s,
                                                                       dim)

            ret['class_sample_densities_train'] = class_sample_densities_train
            ret['class_sample_densities_test'] = class_sample_densities_test
            ret['train_patch_stats'] = train_patch_stats
            ret['test_patch_stats'] = test_patch_stats
            ret['train_log'] = train_output_log
            ret['test_log'] = test_output_log
            return ret

        except:
            print(f'Error running gpstuff.lgpdens on Matlab engine.')
            traceback.print_exc()
            raise
        finally:
            eng.quit()

    @staticmethod
    def ndhistogram(cfg, prev_outputs):
        print(f'Step: Density estimation by histogram ({cfg})')
        n_bins_per_dim = cfg['n_bins_per_dim']
        extreme_bins = cfg['extreme_bins']
        class_train_datasets = prev_outputs['patch_dataset']['class_train_datasets']
        class_test_datasets = prev_outputs['patch_dataset']['class_test_datasets']
        n_classes = len(class_train_datasets)
        extreme_bins = (extreme_bins[0], extreme_bins[1]), (extreme_bins[2], extreme_bins[3])
        step = ((extreme_bins[0][1] - extreme_bins[0][0]) / n_bins_per_dim,
                (extreme_bins[1][1] - extreme_bins[1][0]) / n_bins_per_dim)

        vh_edges = list(np.arange(extreme_bins[0][0], extreme_bins[0][1] + step[0], step[0]))
        vv_edges = list(np.arange(extreme_bins[1][0], extreme_bins[1][1] + step[1], step[1]))

        edges = np.array([vh_edges, vv_edges])
        class_train_histograms = [PatchDatasetBandHistograms(class_train_datasets[i], normalized=False, bins=edges).patch_histograms for i in range(n_classes)]
        class_test_histograms = [PatchDatasetBandHistograms(class_test_datasets[i], normalized=False, bins=edges).patch_histograms for i in range(n_classes)]
        flat_class_train_histograms = [normalize(np.array([class_train_histograms[class_ix][patch_ix, :, :].ravel()
                                                           for patch_ix in range(class_train_histograms[class_ix].shape[0])]))
                                       for class_ix in range(n_classes)]

        flat_class_test_histograms = [normalize(np.array([class_test_histograms[class_ix][patch_ix, :, :].ravel()
                                                          for patch_ix in range(class_test_histograms[class_ix].shape[0])]))
                                      for class_ix in range(n_classes)]

        ret = dict(
            class_sample_densities_train=flat_class_train_histograms,
            class_sample_densities_test=flat_class_test_histograms,
            edges=edges,
            vh_edges=vh_edges,
            vv_edges=vv_edges
        )

        if cfg['estimate_class_densities']:
            print('Estimating train set class-wise densities w/ histograms...')
            class_densities =[]
            for class_ix in range(len(class_train_datasets)):
                X_train = np.vstack([class_train_datasets[class_ix].patch_data[i].band_pixels
                                     for i in range(len(class_train_datasets[class_ix].patch_data))])
                ndhistogram, edges = np.histogramdd(X_train, bins=edges)
                class_densities.append(ndhistogram)
            ret['class_densities'] = class_densities


        return ret

    @staticmethod
    def _item_cache_hash_key(metadata, allowed_keys):
        return str({k: v for (k, v) in metadata.items() if k in allowed_keys})

    @classmethod
    def lgpde_deferred(cls, cfg, prev_outputs):
        deferred_parcel_ids = prev_outputs['patch_dataset']['deferred_parcel_ids']

        merged_patches = []
        for dataset_path in cfg['datasets']:
            print(f'Reading {dataset_path} ...')
            with open(dataset_path, 'rb') as f:
                class_datasets = pickle.load(f)
                for class_dataset in class_datasets:
                    patch_filter = pixel_sampler.include_parcel_ids_filter(deferred_parcel_ids)
                    class_patches = [patch for patch in class_dataset.patch_data if patch_filter(patch)]
                    merged_patches += class_patches

        return cls._sample_lgpde_posterior_to_cache(merged_patches, cfg, prev_outputs)

    @classmethod
    def lgpde(cls, cfg, prev_outputs):
        norm = cfg['normalizer']
        output_raw_samples_to_pipeline_ = cfg['output_raw_samples_to_pipeline']
        do_list_only_cache_misses = 'do_list_only_cache_misses' in cfg and cfg['do_list_only_cache_misses']

        class_train_datasets = prev_outputs['patch_dataset']['class_train_datasets']
        class_test_datasets = prev_outputs['patch_dataset']['class_test_datasets']
        n_train_samples = len(class_train_datasets[0].patch_data)
        n_test_samples = len(class_test_datasets[0].patch_data)

        n_classes = len(class_train_datasets)


        train_class_post_f_samples, train_patch_stats, train_fit_msgs, train_cache_misses = \
            cls._sample_lgpde_posterior(cfg,
                                        class_train_datasets,
                                        'train',
                                        prev_outputs)

        if not do_list_only_cache_misses:
            train_densities_post_mean = [np.array([normalize(np.mean(train_class_post_f_samples[class_ix][m]['exp_f'], axis=0).reshape(1,-1), norm=norm).ravel() for m in range(n_train_samples)]) for class_ix in range(n_classes)]
        else:
            train_densities_post_mean =[]

        if 'class_sample_densities_test' in prev_outputs:
            print('Using cached class_sample_densities_test .')
            test_densities_post_mean, test_patch_stats = prev_outputs['class_sample_densities_test'], None
            test_class_post_f_samples = prev_outputs['test_class_post_f_samples'] if not do_list_only_cache_misses else []
            test_fit_msgs = "N/A (using cached densities for test set)"
            test_cache_misses = []
        else:
            test_class_post_f_samples, test_patch_stats, test_fit_msgs, test_cache_misses = \
                cls._sample_lgpde_posterior(cfg,
                                            class_test_datasets,
                                            'test',
                                            prev_outputs)


            if not do_list_only_cache_misses:
                test_densities_post_mean = [np.array([normalize(np.mean(test_class_post_f_samples[class_ix][m]['exp_f'], axis=0).reshape(1,-1), norm=norm).ravel() for m in range(n_test_samples)]) for class_ix in range(n_classes)]
            else:
                test_densities_post_mean =[]

        ret = dict(
            class_sample_densities_train=train_densities_post_mean,
            class_sample_densities_test=test_densities_post_mean,
            train_patch_stats=train_patch_stats,
            test_patch_stats=test_patch_stats,
            train_fit_msgs=train_fit_msgs,
            test_fit_msgs=test_fit_msgs,
            train_cache_misses=train_cache_misses,
            test_cache_misses=test_cache_misses
        )

        if output_raw_samples_to_pipeline_:
            ret = {
                **ret,
                **dict(
                    train_class_post_f_samples=train_class_post_f_samples,
                    test_class_post_f_samples=test_class_post_f_samples
                )}

        return ret

    @classmethod
    def _sample_lgpde_posterior_to_cache(cls, patches, cfg, prev_outputs):
        model = cls._get_stan_model()

        lgpde_params = M, Sigma, bins, n_chains, n_iter, offsets_vh_vv, params = cls._parse_lgpde_params(cfg)
        print(f'Sampling {len(patches)} parcels with parameters {params}...')
        output_file_list = []
        for sample_ix, patch in tqdm(enumerate(patches)):
            metadata, n_sample, patch_stats, y = \
                cls._extract_patch_metadata_lgpde(lgpde_params, 'cached', cfg, None, patch)

            try:
                with suppress_stdout_stderr('/tmp/lgpde_pystan_stdout.log', '/tmp/lgpde_pystan_stderr.log'):
                    extracts, sampling_log = sample_lgp_posterior(y, M, n_sample, Sigma, model, n_iter,
                                                                  n_chains)  # , mu=hyperprior_mean * np.max(y)
                f_pmu = extract_pmu_stan(extracts)
                posterior_samples_with_metadata = copy.deepcopy(metadata)
                posterior_samples_with_metadata['sampling_output'] = sampling_log,
                posterior_samples_with_metadata['posterior_mean'] = f_pmu,
                posterior_samples_with_metadata['extracts'] = extracts

                raw_samples_dump_dir = f"{prev_outputs['output_dir']}/lgpde_mcmc_post_samples"
                dump_file_path = _make_sample_dump_path(raw_samples_dump_dir, posterior_samples_with_metadata)

                with bz2.BZ2File(dump_file_path, 'wb') as f:
                    pickle.dump(posterior_samples_with_metadata, f)

                output_file_list.append(dump_file_path)
            except:
                print(f'Error on sample. Metadata: {metadata}')
                traceback.print_exc()

            return output_file_list

    @classmethod
    def _sample_lgpde_posterior(cls, cfg, class_datasets, partition_descr, prev_outputs):
        cache_raw_samples_to_disk_ = cfg['cache_raw_samples_to_disk']
        do_list_only_cache_misses = 'do_list_only_cache_misses' in cfg and cfg['do_list_only_cache_misses']
        global_cache_index = prev_outputs['global_cache_index']
        batch_cache_index = prev_outputs['batch_cache_index']
        n_items = len(class_datasets[0].patch_data)
        model = cls._get_stan_model()

        lgpde_params = M, Sigma, bins, n_chains, n_iter, offsets_vh_vv, params = cls._parse_lgpde_params(cfg)

        print(f'Proceeding to sample {partition_descr} set of {n_items} items...')
        class_post_f_samples = []
        sampling_msgs = []
        classes_patch_stats = []
        cache_misses = []

        for class_ix, samples in tqdm(enumerate(class_datasets)):
            post_f_samples = []
            n_cache_hits = 0
            class_patch_stats = []
            for sample_ix in tqdm(range(len(samples.patch_data))):
                patch = class_datasets[class_ix].patch_data[sample_ix]

                metadata, n_sample, patch_stats, y = \
                    cls._extract_patch_metadata_lgpde(lgpde_params, partition_descr, cfg, class_ix, patch)

                cache_key = DensityEstimators._item_cache_hash_key(metadata, _LGPDE_CACHE_KEYS)
                cache_hit = False
                if do_list_only_cache_misses:
                    if cache_key not in global_cache_index:
                        cache_misses.append(metadata)
                else:
                    try:
                        if cache_key in global_cache_index:
                            n_cache_hits = n_cache_hits + 1
                            # print(f"Loading patch representation from cache: {cache_key}, loading \n\t{global_cache_index[cache_key]['dump_file_path']} ...")
                            dump_file_path = global_cache_index[cache_key]['dump_file_path']
                            with bz2.BZ2File(dump_file_path, 'rb') as f:
                                posterior_samples_with_metadata = pickle.load(f)
                                sampling_log = posterior_samples_with_metadata['sampling_output']
                                f_pmu = posterior_samples_with_metadata['posterior_mean']
                                extracts = posterior_samples_with_metadata['extracts']
                                cache_hit = True
                        else:
                            with suppress_stdout_stderr('/tmp/lgpde_pystan_stdout.log', '/tmp/lgpde_pystan_stderr.log'):
                                extracts, sampling_log = sample_lgp_posterior(y, M, n_sample, Sigma, model, n_iter,
                                                                               n_chains)  # , mu=hyperprior_mean * np.max(y)
                            f_pmu = extract_pmu_stan(extracts)
                            posterior_samples_with_metadata = copy.deepcopy(metadata)
                            posterior_samples_with_metadata['sampling_output'] = sampling_log,
                            posterior_samples_with_metadata['posterior_mean'] = f_pmu,
                            posterior_samples_with_metadata['extracts'] = extracts

                        if cache_raw_samples_to_disk_ and not cache_hit:
                            raw_samples_dump_dir = f"{prev_outputs['output_dir']}/lgpde_mcmc_post_samples"
                            dump_file_path = _make_sample_dump_path(raw_samples_dump_dir, posterior_samples_with_metadata)

                            with bz2.BZ2File(dump_file_path, 'wb') as f:
                                pickle.dump(posterior_samples_with_metadata, f)

                            batch_cache_index[cache_key] = \
                                dict(metadata=metadata, dump_file_path=dump_file_path)

                        post_f_samples.append(extracts)
                        sampling_msgs.append(sampling_log)
                        class_patch_stats.append(patch_stats)
                    except:
                        print(f'Error handling class {class_ix} on {partition_descr} sample {sample_ix}')
                        traceback.print_exc()
                        post_f_samples.append(None)
            if not do_list_only_cache_misses:
                print(f"Found and used {n_cache_hits} cached precomputed patch densities out of {len(samples.patch_data)} "
                      f"for {partition_descr} set of class {class_ix}.")
                class_post_f_samples.append(post_f_samples)
                classes_patch_stats.append(class_patch_stats)
        print(len(cache_misses))
        return class_post_f_samples, classes_patch_stats, sampling_msgs, cache_misses

    @classmethod
    def _extract_patch_metadata_lgpde(cls, lgpde_params, partition_descr, cfg, class_ix, patch):
        _, Sigma, bins, _, _, offsets_vh_vv, params = lgpde_params
        patch_stats = {'parcelID': patch.patch_properties['parcelID'],
                       'n_pixels': patch.band_pixels.shape[0]}
        X = patch.band_pixels
        n_sample = X.shape[0]
        ndhistogram, edges = np.histogramdd(X + offsets_vh_vv, bins=bins)
        y = ndhistogram.ravel()
        metadata = dict(
            step_name='lgpde',
            partition_descr=partition_descr,
            patch_id=patch.patch_properties['parcelID'],
            class_ix=class_ix,
            n_pixels=patch.band_pixels.shape[0],
            n_bins_per_dim=cfg['n_bins_per_dim'],
            extreme_bins=cfg['extreme_bins'],
            params=params,
            prior_sigma=Sigma,
        )
        return metadata, n_sample, patch_stats, y

    @classmethod
    def _parse_lgpde_params(cls, cfg):
        params = cfg['params']
        m = cfg['n_bins_per_dim']
        bsc_min_vh, bsc_max_vh, bsc_min_vv, bsc_max_vv = cfg['extreme_bins']
        n_iter = params['n_iter']
        n_chains = params['n_chains']
        h_basis, dim_basis = LGP_BASIS[params['basis']]
        beta = params['beta']
        length_scale = params['length_scale']
        M = m ** 2  # num of normally distributed variables
        d_vh, d_vv = (bsc_max_vh - bsc_min_vh) / m, (bsc_max_vv - bsc_min_vv) / m
        offset_vh, offset_vv = (bsc_max_vh - bsc_min_vh) / 2, (bsc_max_vv - bsc_min_vv) / 2
        offsets_vh_vv = np.array([offset_vh, offset_vv]).reshape(1, -1)
        vh_edges = np.arange(bsc_min_vh, bsc_max_vh + d_vh, d_vh) + offset_vh
        vv_edges = np.arange(bsc_min_vv, bsc_max_vv + d_vv, d_vv) + offset_vv
        vh_test, vv_test = np.meshgrid(vh_edges[:-1], vv_edges[:-1])
        X_s = np.vstack([vh_test.ravel(), vv_test.ravel()]).T
        bins = np.array([vh_edges, vv_edges])
        Sigma = lgp_cov_prior(h_basis, dim_basis, X_s, l=length_scale, beta=beta)
        return M, Sigma, bins, n_chains, n_iter, offsets_vh_vv, params
