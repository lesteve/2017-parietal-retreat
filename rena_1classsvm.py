from scipy.stats import kurtosis
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, AgglomerativeClustering
from nilearn.image import resample_img
from nilearn.plotting import plot_epi
from nilearn import datasets
from nilearn.input_data import NiftiMasker
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../ReNA')
from rena import ReNA
from utils import load_data


def get_dummy_features(X):
    feats = [np.mean, np.max, np.min, np.std, kurtosis]
    return np.concatenate([feat(X, axis=1)[:, None] for feat in feats],
                          axis=1)


def get_data_by_indices(masker, neurovault_data, indices):
    """Return neurovault data with specified indices, preprocessed with a
       MNI 152 mask.
    """
    X_ = []
    for ix in indices:
        file_name = neurovault_data["images"][ix]
        X_.append(masker.transform(file_name))

    return np.concatenate(X_)


def get_data_by_names(masker, names):
    """Return neurovault data with specified indices, preprocessed with a
       MNI 152 mask.
    """
    X_ = []
    for file_name in names:
        X_.append(masker.transform(file_name))

    return np.concatenate(X_)


if __name__ == "__main__":
    image_terms = {"not_mni": False, "is_valid": True}
    neurovault_data = load_data(verbose=3, image_terms=image_terms)

    neurovault_data.images = np.array(neurovault_data.images)
    n_samples = len(neurovault_data.images)

    target_img = datasets.load_mni152_brain_mask()
    target_img = resample_img(target_img, target_affine=np.diag([4, 4, 4]),
                              interpolation='nearest')
    masker = NiftiMasker(
        mask_img=target_img, smoothing_fwhm=2, standardize=False,
        target_affine=target_img.affine, target_shape=target_img.shape,
        memory='nilearn_cache')
    masker = masker.fit()

    # 1st part: get dummy features and train a OCSVM on it to eliminate obvious
    # outliers, so as not to feed garbage to ReNA
    already_saved = True
    if already_saved:
        means = np.zeros(n_samples)
        maxs = np.zeros(n_samples)
        mins = np.zeros(n_samples)
        stddevs = np.zeros(n_samples)
        kurts = np.zeros(n_samples)

        slice_size = 500
        for slice_start in range(0, n_samples, slice_size):
            print("Processing slice starting at %d" % slice_start)
            slice_end = min(slice_start + slice_size, n_samples)
            indices = range(slice_start, slice_end)
            X = get_data_by_indices(masker, neurovault_data, indices)
            means[indices] = np.mean(X, axis=1)
            maxs[indices] = np.max(X, axis=1)
            mins[indices] = np.min(X, axis=1)
            stddevs[indices] = np.std(X, axis=1)
            kurts[indices] = kurtosis(X, axis=1)

        np.save("means", means)
        np.save("maxs", maxs)
        np.save("mins", mins)
        np.save("stddevs", stddevs)
        np.save("kurts", kurts)
    else:
        means = np.load("means.npy")
        maxs = np.load("maxs.npy")
        mins = np.load("mins.npy")
        stddevs = np.load("stddevs.npy")
        kurts = np.load("kurts.npy")

    X_data = np.concatenate([means[:, None], maxs[:, None], mins[:, None],
                             stddevs[:, None], kurts[:, None]],
                            axis=1)
    # first step: exclude data with nan features:
    exclude = np.isnan(X_data).any(axis=1)
    X_data_filtered = X_data[~exclude]
    inds_filtered = np.arange(n_samples)[~exclude]
    names_filtered = neurovault_data.images[~exclude]
    clf = OneClassSVM(nu=0.3, kernel="linear")
    clf.fit(X_data_filtered)

    # sanity check :we know that the data from this collection is clean
    # because it was uploaded by Bertrand
    clean = datasets.fetch_neurovault(max_images=None, mode='offline',
                                      image_terms={'collection_id': 656})
    X_bertrand = get_data_by_names(masker, clean.images)
    # the following fails if nu is too high
    np.testing.assert_equal(clf.predict(get_dummy_features(X_bertrand)), 1.)

    pred = clf.predict(X_data_filtered)
    # getting back the indices of the outliers in the non filtered data:
    outliers = inds_filtered[pred != 1.]
    print("# of outliers: %d" % np.sum(pred != 1.))

    # X_outliers = get_data_by_indices(neurovault_data, outliers)
    #

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
    ind_train_rena = np.random.choice(valid, n_samples_rena, replace=False)
    X_train_rena = get_data_by_indices(masker, neurovault_data, ind_train_rena)

    # X_train_data should be only composed of valid data, however the following
    # fails:
    np.testing.assert_equal(clf.predict(get_dummy_features(X_train_rena)),
                            1.)

    n_dims_reduced = 2000  # arbitrary
    cluster = ReNA(masker=masker, n_clusters=n_dims_reduced)
    cluster.fit(X_train_rena)
    print("%.1f %% of pop. in the largest cluster" % (
          100 * np.max(np.bincount(cluster.labels_)) / cluster.labels_.shape))

    visualize_examples = True
    plt.close('all')
    if visualize_examples:
        X_reduced = cluster.transform(X_train_rena[0: 10])
        X_compressed = cluster.inverse_transform(X_reduced)

        cut_coords = (-34, -16)
        for n_img in range(10):
            compress_fig = plot_epi(masker.inverse_transform(X_compressed[n_img]),
                                    title='compressed', display_mode='yz',
                                    cut_coords=cut_coords)

            original_fig = plot_epi(masker.inverse_transform(X_train_rena[n_img]),
                                    title='original', display_mode='yz',
                                    cut_coords=cut_coords)

            compress_fig.savefig('fig_temp/%d_compress.png' % n_img)
            original_fig.savefig('fig_temp/%d_original.png' % n_img)
    # plt.show()

    n_samples_ward = 3000
    ind_train_ward = np.random.choice(valid, n_samples_ward, replace=False)
    X_train_ward = get_data_by_indices(masker, neurovault_data, ind_train_ward)
    # fit ward on transpose ?
    ward = AgglomerativeClustering(n_clusters=2000).fit(X_train_ward.T)



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

    # KMeans only on valid data
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(
        X_reduced[valid])
    labels = kmeans.labels_  # this gives 9 clusters with population < 4

    populations = np.bincount(labels)
    plt.close('all')
    for ix in range(n_clusters):
        if populations[ix] < 5:
            this_cluster = valid[labels == ix]
            for i, img in enumerate(X_reduced[this_cluster]):
                img_compressed = cluster.inverse_transform(img)
                compress_fig = plot_epi(masker.inverse_transform(img_compressed),
                                        title='compressed', display_mode='yz',
                                        cut_coords=cut_coords)

                img_original = get_data_by_names([neurovault_data.images[this_cluster[i]]])[0]
                original_fig = plot_epi(masker.inverse_transform(img_original),
                                        title='original', display_mode='yz',
                                        cut_coords=cut_coords)

                compress_fig.savefig('fig_kmeans/cluster%d_%d_compress.png' %
                                     (ix, i))
                original_fig.savefig('fig_kmeans/cluster%d_%d_original.png' %
                                     (ix, i))

    plt.show()
