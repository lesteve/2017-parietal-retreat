import numpy as np
from nilearn import datasets
from nilearn.input_data import NiftiMasker


def load_data(verbose=0):
    """Fetch neurovault data, filtering out non MNI images.
    """
    neurovault_data = datasets.fetch_neurovault(
        max_images=None, mode="offline", verbose=verbose,
        image_terms={"not_mni": False})
    return neurovault_data


def preprocess_data(neurovault_data, n_samples):
    """Return the first n_samples of neurovault, preprocessed with a
       MNI 152 mask.
    """
    target_img = datasets.load_mni152_brain_mask()
    masker = NiftiMasker(
        mask_img=target_img, smoothing_fwhm=2, standardize=False,
        target_affine=target_img.affine, target_shape=target_img.shape,
        memory='nilearn_cache')
    masker = masker.fit()

    X_ = []
    for file_name in neurovault_data["images"][:n_samples]:
        X_.append(masker.transform(file_name))

    return np.concatenate(X_)
