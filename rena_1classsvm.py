from utils import load_data
from scipy.stats import kurtosis
from sklearn.svm import OneClassSVM
from nilearn.plotting import plot_epi
from nilearn import datasets
from nilearn.input_data import NiftiMasker
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../ReNA')
from rena import ReNA


def get_data_by_indices(neurovault_data, indices):
    """Return neurovault data with specified indices, preprocessed with a
       MNI 152 mask.
    """
    target_img = datasets.load_mni152_brain_mask()
    masker = NiftiMasker(
        mask_img=target_img, smoothing_fwhm=2, standardize=False,
        target_affine=target_img.affine, target_shape=target_img.shape,
        memory='nilearn_cache')
    masker = masker.fit()

    X_ = []
    for ix in indices:
        file_name = neurovault_data["images"][ix]
        X_.append(masker.transform(file_name))

    return np.concatenate(X_)


def get_data_by_names(names):
    """Return neurovault data with specified indices, preprocessed with a
       MNI 152 mask.
    """
    target_img = datasets.load_mni152_brain_mask()
    masker = NiftiMasker(
        mask_img=target_img, smoothing_fwhm=2, standardize=False,
        target_affine=target_img.affine, target_shape=target_img.shape,
        memory='nilearn_cache')
    masker = masker.fit()

    X_ = []
    for file_name in names:
        X_.append(masker.transform(file_name))

    return np.concatenate(X_)


if __name__ == "__main__":
    bads = set()
    image_terms = {"not_mni": False, "is_valid": True}
    neurovault_data = load_data(verbose=3, image_terms=image_terms)

    neurovault_data.images = np.array(neurovault_data.images)
    n_samples = len(neurovault_data.images)

    # 1st part: get dummy features and train a OCSVM on it to eliminate obvious
    # outliers, so as not to feed garbage to ReNA
    means = np.zeros(n_samples)
    maxs = np.zeros(n_samples)
    mins = np.zeros(n_samples)
    stddevs = np.zeros(n_samples)
    kurts = np.zeros(n_samples)

    slice_size = 500
    for slice_start in range(0, n_samples, slice_size):
        print("Processing slice starting at %d" % slice_start)
        indices = range(slice_start, min(slice_start + slice_size, n_samples))
        X = get_data_by_indices(neurovault_data, indices)
        means[slice_start: min(slice_start + slice_size, n_samples)] = np.mean(X, axis=1)
        maxs[slice_start: min(slice_start + slice_size, n_samples)] = np.max(X, axis=1)
        mins[slice_start: min(slice_start + slice_size, n_samples)] = np.min(X, axis=1)
        stddevs[slice_start: min(slice_start + slice_size, n_samples)] = np.std(X, axis=1)
        kurts[slice_start: min(slice_start + slice_size, n_samples)] = kurtosis(X, axis=1)

    np.save("means", means)
    np.save("maxs", maxs)
    np.save("mins", mins)
    np.save("stddevs", stddevs)
    np.save("kurts", kurts)

    X_data = np.concatenate([means[:, None], maxs[:, None], mins[:, None],
                             stddevs[:, None], kurts[:, None]],
                            axis=1)
    exclude = np.isnan(X_data).any(axis=1)
    X_data_filtered = X_data[~exclude]
    inds_filtered = np.arange(n_samples)[~exclude]
    names_filtered = neurovault_data.images[~exclude]
    clf = OneClassSVM(nu=0.01, kernel="linear")
    clf.fit(X_data_filtered)
    pred = clf.predict(X_data_filtered)
    # getting back the indices of the outliers in the non filtered data:
    outliers = inds_filtered[np.where(pred != 1.)[0]]
    print("# of outliers: %d" % np.sum(pred != 1.))

    # X_outliers = get_data_by_indices(neurovault_data, outliers)
    #
    target_img = datasets.load_mni152_brain_mask()
    masker = NiftiMasker(
        mask_img=target_img, smoothing_fwhm=2, standardize=False,
        target_affine=target_img.affine, target_shape=target_img.shape,
        memory='nilearn_cache')
    masker = masker.fit()
    #
    # cut_coords = (-34, -16)
    # for outlier in X_outliers[10:20]:
    #     compress_fig = plot_epi(masker.inverse_transform(outlier),
    #                             title='outlier', display_mode='yz',
    #                             cut_coords=cut_coords)
    # plt.show()

    # train ReNA on valid data
    valid = inds_filtered[pred == 1]
    n_samples_rena = 500
    ind = np.random.choice(valid, n_samples_rena, replace=False)
    X = get_data_by_indices(neurovault_data, ind)

    n_dims_reduced = 3000  # arbitrary
    cluster = ReNA(masker=masker, n_clusters=n_dims_reduced)
    cluster.fit(X)

    visualize_examples = False
    if visualize_examples:
        X_reduced = cluster.transform(X[0: 10])
        X_compressed = cluster.inverse_transform(X_reduced)

        cut_coords = (-34, -16)
        for n_image in range(10):
            compress_fig = plot_epi(masker.inverse_transform(X_compressed[n_image]),
                                    title='compressed', display_mode='yz',
                                    cut_coords=cut_coords)

            original_fig = plot_epi(masker.inverse_transform(X[n_image]),
                                    title='original', display_mode='yz',
                                    cut_coords=cut_coords)

            compress_fig.savefig('figures_valid/%d_compress.png' % n_image)
            original_fig.savefig('figures_valid/%d_original.png' % n_image)

    # get all compressed data
    X_reduced = np.zeros((n_samples, n_dims_reduced))

    slice_size = 500
    for slice_start in range(0, n_samples, slice_size):
        print("Processing slice starting at %d" % slice_start)
        slice_end = min(slice_start + slice_size, n_samples)
        indices = range(slice_start, slice_end)
        X = get_data_by_indices(neurovault_data, indices)
        X_reduced[slice_start: slice_end] = cluster.transform(X)
    np.save("X_reduced.npy", X_reduced)


    dd = datasets.fetch_neurovault(max_images=None, mode='offline',
                                   image_terms={'collection_id': 656})
    X_bertrand = get_data_by_names(dd.images)
