import sys

import numpy as np
import pandas as pd
import sklearn.preprocessing as sklpp
from sklearn.preprocessing import normalize


# These classifier imports are not directly referred to in code, instead from configuration. DO NOT CLEAN OUT.
# Add as needed for more classifiers.

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from dataset_reader import valid_density_dataset


EPS = 0.00000001


def _str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def classifier_sklearn(X_train, X_test, y, classifier_name, params):
    """

    :param X_train:
    :param X_test:
    :param y:
    :param classifier_name: class must be imported at top of the file, for config use class name only
    :param params:
    :return:
    """
    n_classes = np.unique(y).shape[0]
    classifier = _str_to_class(classifier_name)
    clf = classifier(**params)
    clf.fit(X_train, y.ravel())
    predictions = [clf.predict(hists) for hists in X_test]
    cm = pd.DataFrame(np.array([[np.sum(predictions[real_class_ix] == test_class_ix) / predictions[real_class_ix].shape[0]
                                 for test_class_ix in range(n_classes)]
                                for real_class_ix in range(n_classes)]))
    return cm


def _kl_divergence(p, q):
    return np.sum(p * (np.log(p + EPS) - np.log(q+EPS)), axis=1)


def similarity_kl(norm_class_densities, norm_sample_densities):
    return [np.hstack([-_kl_divergence(norm_class_densities[test_class_ix], norm_sample_densities[real_class_ix]).reshape(-1,1)
                       for test_class_ix in range(len(norm_sample_densities))])
            for real_class_ix in range(len(norm_sample_densities))]


def similarity_cosine(norm_class_densities, norm_sample_densities):
    """
    Assume flattened, L2-normalized input densities.
    """
    n_classes = len(norm_class_densities)
    return [(norm_sample_densities[real_class_ix] @ np.vstack(norm_class_densities).T)
            for real_class_ix in range(n_classes)]


def similarity_euclidean(norm_class_densities, norm_sample_densities):
    return [np.hstack([np.linalg.norm(norm_class_densities[test_class_ix] - norm_sample_densities[real_class_ix], axis=1).reshape(-1,1)
                       for test_class_ix in range(len(norm_sample_densities))])
            for real_class_ix in range(len(norm_sample_densities))]

NN_DISTANCE_MEASURES = {
    'similarity_cosine': similarity_cosine,
    'similarity_euclidean': similarity_euclidean,
    'similarity_kl': similarity_kl
}


def total_accuracy(cm, test_set_size):
    acc = np.sum(np.diag(cm) * test_set_size) / np.sum(test_set_size)
    print(f'Class accuracies: {np.diag(cm).tolist()}')
    print('Total accuracy: %.3f' % acc)
    return acc


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


def subset_posterior_patch_samples(samples_arr, q=0.5, n=500):
    class_data = []
    n_classes = samples_arr.shape[0]
    for class_ix in range(n_classes):
        data = samples_arr[class_ix,:,:,1:]
        quantile_limit = np.quantile(samples_arr[class_ix,:,:,0], 1-q, axis=1)
        ix = np.moveaxis(np.moveaxis(samples_arr[class_ix, :, :, 0], 1,0) > quantile_limit, 1, 0)
        data = data[ix]
        downsample_ix = np.random.choice(data.shape[0], n, replace=True)
        class_data.append(data[downsample_ix])
    y = np.array(list([class_ix for class_ix in range(n_classes) for m in range(n)]))
    return class_data, y


class Classifiers:
    @staticmethod
    def nearest_neighbor_classifier(classifier_config, prev_outputs):
        print(f"Step: Nearest neighbor classifier ({classifier_config['nearest_neighbor_distance']})")
        cm, test_set_size = Classifiers._nearest_neighbor_classification(classifier_config, prev_outputs)
        acc = total_accuracy(cm, test_set_size)

        return dict(
            cm=cm,
            acc=acc)

    @staticmethod
    def sklearn_classifier(cfg, prev_outputs):
        print(f"Step: SKLEARN classifier ({cfg['classifier']})({cfg})")
        class_sample_densities_train = prev_outputs['density_estimate']['class_sample_densities_train']
        class_sample_densities_test = prev_outputs['density_estimate']['class_sample_densities_test']
        norm = cfg['normalizer']
        params = cfg['params']

        cm = Classifiers._sklearn_classification(class_sample_densities_test,
                                                 class_sample_densities_train,
                                                 cfg['classifier'],
                                                 norm,
                                                 params)

        test_set_size = np.array([len(class_data) for class_data in class_sample_densities_test])
        acc = total_accuracy(cm, test_set_size)

        return dict(cm=cm, acc=acc)


    @staticmethod
    def sklearn_classifier_full_bayes(cfg, prev_outputs):
        print(f"Step: SKLEARN classifier ({cfg['classifier']})({cfg})")
        class_sample_densities_train = prev_outputs['density_estimate']['class_sample_densities_train']
        class_sample_densities_test = prev_outputs['density_estimate']['class_sample_densities_test']
        test_class_post_f_samples = prev_outputs['density_estimate']['test_class_post_f_samples']
        norm = cfg['normalizer']
        params = cfg['params']

        cm, predictions = Classifiers._sklearn_classification_full_bayes(test_class_post_f_samples,
                                                            class_sample_densities_train,
                                                            cfg['classifier'],
                                                            norm,
                                                            params)

        test_set_size = np.array([len(class_data) for class_data in class_sample_densities_test])
        acc = total_accuracy(cm, test_set_size)

        return dict(cm=cm, acc=acc, predictions=predictions)


    @staticmethod
    def _nearest_neighbor_classification(classifier_config, prev_outputs):
        comparison_measure = NN_DISTANCE_MEASURES[classifier_config['nearest_neighbor_distance']]
        class_sample_densities_train = prev_outputs['density_estimate']['class_sample_densities_train']
        class_sample_densities_test = prev_outputs['density_estimate']['class_sample_densities_test']
        class_densities = prev_outputs['density_estimate']['class_densities']
        auxiliary_data_test = prev_outputs['patch_dataset']['auxiliary_data_test']
        norm = classifier_config['normalizer']
        n_classes = len(class_sample_densities_train)
        test_set_size = np.array([len(class_data) for class_data in class_sample_densities_test])
        class_sample_densities_test, auxiliary_data_test = valid_density_dataset(class_sample_densities_test,
                                                                                 auxiliary_data_test)
        cm = Classifiers._nn_classify_with_metrics(class_densities, class_sample_densities_test, comparison_measure,
                                                   n_classes, norm)
        return cm, test_set_size

    @staticmethod
    def _nn_classify_with_metrics(class_densities, class_sample_densities_test, comparison_measure, n_classes, norm):
        # TODO: configurable normalization!
        norm_class_densities = np.array(
            [normalize(class_densities[class_ix].ravel().reshape(1, -1), norm=norm) for class_ix in range(n_classes)])
        norm_sample_densities = []
        for class_ix in range(n_classes):
            norm_sample_densities.append(np.vstack(
                [normalize(s.ravel().reshape(1, -1)) for s in class_sample_densities_test[class_ix] if
                 np.all(np.isfinite(s))]))
        class_membership = comparison_measure(norm_class_densities, norm_sample_densities)
        cm = pd.DataFrame(np.array([[np.sum(np.argmax(class_membership[real_class_ix], axis=1) == test_class_ix)
                                     / class_membership[real_class_ix].shape[0]
                                     for test_class_ix in range(n_classes)] for real_class_ix in range(n_classes)]))
        return cm

    @staticmethod
    def _sklearn_classification(class_sample_densities_test, class_sample_densities_train, classifier_name, norm,
                                params):
        X_train = np.vstack(class_sample_densities_train)
        y = np.vstack([np.array([class_ix] * hists.shape[0]).reshape(-1, 1) for class_ix, hists in
                       enumerate(class_sample_densities_train)]).ravel()
        valid_ix_train = [np.all(np.isfinite(row)) for row in X_train]
        X_train.shape, y.shape, len(valid_ix_train), np.count_nonzero(valid_ix_train)
        X_test = [sklpp.normalize(densities[np.isfinite(densities[:, 0])], norm=norm) for class_ix, densities in
                  enumerate(class_sample_densities_test)]
        cm = classifier_sklearn(sklpp.normalize(X_train[valid_ix_train, :], norm=norm, axis=1),
                                X_test,
                                y[valid_ix_train],
                                classifier_name,
                                params)
        return cm

    @staticmethod
    def _sklearn_classification_full_bayes(test_class_post_f_samples, class_sample_densities_train, classifier_name,
                                           norm, params):
        X_train = np.vstack(class_sample_densities_train)
        y = np.vstack([np.array([class_ix] * hists.shape[0]).reshape(-1, 1) for class_ix, hists in
                       enumerate(class_sample_densities_train)]).ravel()
        valid_ix_train = [np.all(np.isfinite(row)) for row in X_train]
        X_train.shape, y.shape, len(valid_ix_train), np.count_nonzero(valid_ix_train)

        train = sklpp.normalize(X_train[valid_ix_train, :], norm=norm, axis=1)
        y1 = y[valid_ix_train]
        n_classes = np.unique(y1).shape[0]
        classifier = _str_to_class(classifier_name)
        clf = classifier(**params)
        clf.fit(train, y1.ravel())
        X_test_samples = posterior_patch_samples_as_array(test_class_post_f_samples)
        n_samples_per_patch = 100 # todo move to config params
        n_test_patches = X_test_samples.shape[1]
        n_samples_per_patch_max = X_test_samples.shape[2]
        subset_index = np.array([[[np.random.choice(n_samples_per_patch_max, n_samples_per_patch, replace=False)
                                   for m in range(n_test_patches)]
                                  for n in range(n_classes)]
                                 for o in range(145)])
        subset_index = np.moveaxis(subset_index, 0, -1)
        X_test_samples = np.take_along_axis(X_test_samples, subset_index, 2)[:,:,:,1:]
        test_set_shape = list(X_test_samples.shape)
        predictions = \
            clf.predict(
                sklpp.normalize(
                    X_test_samples.reshape(-1, test_set_shape[-1]),
                    norm='l2',
                    axis=1)
            ).reshape(*(test_set_shape[:-1]))

        predictions = np.moveaxis(np.concatenate([[np.sum(predictions == class_ix, axis=-1)] for class_ix in range(3)]), 0, -1)
        predictions_shape = predictions.shape
        predictions = sklpp.normalize(predictions.reshape(-1, predictions_shape[-1]), axis=1, norm='l1').reshape(predictions_shape)
        cm = pd.DataFrame(np.array([[np.sum(np.argmax(predictions[real_class_ix], axis=-1) == test_class_ix) / predictions[real_class_ix].shape[0]
                                      for test_class_ix in range(n_classes)]
                                     for real_class_ix in range(n_classes)]))
        print(cm)
        return cm, predictions

    def mlp_classifier(classifier_config, input_):
        print(f'Step: mlp classifier ({classifier_config})')
        return None


