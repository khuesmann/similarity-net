import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def calculate_field_dissimilarity(sample_i, samples):
    distance_matrix = []
    for j, sample_j in enumerate(samples):
        maxi = np.amax(np.array([sample_i, sample_j]), axis=0)
        mini = np.amin(np.array([sample_i, sample_j]), axis=0)
        d_AB = 1. - np.sum(1 - maxi) / np.sum(1 - mini)
        # if np.isinf(d_AB):
        #     d_AB = np.mean(distance_matrix)
        distance_matrix.append(d_AB)
    return distance_matrix


def calculate_correlation_dissimilarity(sample_i, samples):
    distance_matrix = []
    for j, sample_j in enumerate(samples):
        corr = np.corrcoef(sample_i, sample_j)[0, 1]
        d_corr = 1. - ((corr + 1) / 2.)
        distance_matrix.append(d_corr)
    return distance_matrix


def calculate_distance_matrix(runs, method='field_similarity', threads=1):
    distance_matrix = []
    samples = []
    for run in runs:
        for timestep in run:
            if timestep[0] == -1:
                break
            samples.append(timestep)

    if method == 'field_similarity':
        distance_matrix = Parallel(n_jobs=threads)(delayed(calculate_field_dissimilarity)(sample_i, samples) for sample_i in tqdm(samples, desc=f'Distance matrix'))
    elif method == 'correlation':
        distance_matrix = Parallel(n_jobs=threads)(delayed(calculate_correlation_dissimilarity)(sample_i, samples) for sample_i in tqdm(samples, desc=f'Distance matrix'))

    distance_matrix = np.array(distance_matrix)
    print('Created distance matrix ({} x {})'.format(distance_matrix.shape[0], distance_matrix.shape[1]))
    return distance_matrix


def mds(d, dimensions=3, method='cpu'):
    (n, n) = d.shape
    E = (-0.5 * d ** 2)

    Er = np.mat(np.mean(E, 1))
    Es = np.mat(np.mean(E, 0))

    F = np.array(E - np.transpose(Er) - Es + np.mean(E))

    [U, S, V] = np.linalg.svd(F.astype(np.float64), full_matrices=True, hermitian=True)

    Y = U * np.sqrt(S)

    return (Y[:, 0:dimensions], S[0:dimensions])


def calculate_mds(distance_matrix, durations):
    X, eigen = mds(distance_matrix, 10, 'cpu')
    X_splits = []
    prev_i = 0
    for i, duration in enumerate(durations):
        split = X[prev_i:prev_i + duration, :]
        prev_i += duration
        X_splits.append(split)
    return X_splits, eigen
