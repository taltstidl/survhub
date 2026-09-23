import numpy as np
from sksurv.util import check_y_survival


def concordance_index_antolini(survival_test: np.ndarray, estimate: np.ndarray, times: np.ndarray):
    test_event, test_time = check_y_survival(survival_test)
    comp = np.expand_dims(test_event, -2) & ((np.expand_dims(test_time, -2) < np.expand_dims(test_time, -1)) | (
            (np.expand_dims(test_time, -2) == np.expand_dims(test_time, -1)) & ~np.expand_dims(test_event, -1)))

    # Survival probabilities estimate[i] are given at times[i], calculate source indices for test samples
    idx_event = np.searchsorted(times, test_time, side='right') - 1
    idx_censored = np.searchsorted(times, test_time, side='right') - 1
    idx = np.where(test_event, idx_event, idx_censored)
    test_time_idx = np.clip(idx, 0, times.shape[-1] - 1)

    survival_i = np.take_along_axis(estimate, np.expand_dims(test_time_idx, -1), axis=-1).transpose(-1, -2)
    survival_j = np.take_along_axis(estimate, np.expand_dims(test_time_idx, -2), axis=-1)
    conc = (survival_i < survival_j) & comp

    return conc.sum(axis=(-2, -1)) / comp.sum(axis=(-2, -1))
