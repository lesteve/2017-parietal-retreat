import numpy as np
from nilearn import datasets
from nilearn.input_data import NiftiMasker


def load_data(verbose=0, image_terms=None):
    """Fetch neurovault data, filtering out non MNI images.
    """
    if image_terms is None:
        image_terms = {"not_mni": False}
    neurovault_data = datasets.fetch_neurovault(
        max_images=None, mode="offline", verbose=verbose,
        image_terms=image_terms)
    return neurovault_data


def preprocess_data(masker, neurovault_data, n_samples):
    """Return the first n_samples of neurovault, preprocessed with the given
        mask
    """

    X_ = []
    for file_name in neurovault_data["images"][:n_samples]:
        X_.append(masker.transform(file_name))

    return np.concatenate(X_)
