from pixel_sampler import Patch, PatchDataset
import numpy as np


class PatchBandHistogram:

    def __init__(self, patch: Patch, bins: [[float]], range: [(float, float)],
                 synthetic_bands=None, normalized=True, integer_base=None):

        if synthetic_bands:
            bands = np.hstack([patch.band_pixels, synthetic_bands(patch.band_pixels)])
        else:
            bands = patch.band_pixels

        self.n_pixels = patch.band_pixels.shape[0]
        ndhistogram, edges = np.histogramdd(bands, bins=bins, range=range)
        self.ndhistogram = ndhistogram
        self.edges = edges

        if normalized or integer_base:
            self.normalize(integer_base)


    def normalize(self, integer_base=None, rel_to_max=None):
        if integer_base:
            ndhistogram = np.exp(np.log(integer_base) - np.log(self.n_pixels) + np.log(self.ndhistogram)).astype(
                np.int32)
            residue = integer_base - np.sum(ndhistogram)
            ix = np.unravel_index(np.argmax(ndhistogram, axis=None), ndhistogram.shape)
            ndhistogram[ix] += residue
            self.ndhistogram = ndhistogram
        else:
            if np.sum(self.ndhistogram) > 0:
                ndhistogram = self.ndhistogram / np.max(self.ndhistogram)
                self.ndhistogram = ndhistogram


class PatchDatasetBandHistograms:
    def __init__(self, patch_dataset: PatchDataset, bins: [[float]] = None, range: [(float, float)] = None,
                 synthetic_bands=None, normalized=True, integer_base=None):
        self.patch_histograms = np.array([PatchBandHistogram(patch, synthetic_bands=synthetic_bands, bins=bins,
                                                             range=range, normalized=normalized,
                                                             integer_base=integer_base).ndhistogram
                                          for patch in patch_dataset.patch_data])
        print(f'Generated {self.patch_histograms.shape[0]} histograms')
        valid_ix = np.sum(self.patch_histograms.reshape(self.patch_histograms.shape[0], self.patch_histograms.shape[1] * self.patch_histograms.shape[2]), axis=1) > 0
        self.patch_histograms = self.patch_histograms[valid_ix, :, :]
        print(f'Retaining {self.patch_histograms.shape[0]} histograms (others were all-zero)')
