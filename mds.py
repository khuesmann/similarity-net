import os
import pickle

import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


def mds(d, dimensions=3):
    (n, n) = d.shape
    E = (-0.5 * d ** 2)

    Er = np.mat(np.mean(E, 1))
    Es = np.mat(np.mean(E, 0))

    F = np.array(E - np.transpose(Er) - Es + np.mean(E))

    [U, S, V] = np.linalg.svd(F.astype(np.float64), full_matrices=True, hermitian=True)

    Y = U * np.sqrt(S)

    return Y[:, 0:dimensions], S[0:dimensions]


def calculate_mds(distance_matrix, durations):
    X, eigen = mds(distance_matrix, 10)
    X_splits = []
    prev_i = 0
    for i, duration in enumerate(durations):
        split = X[prev_i:prev_i + duration, :]
        prev_i += duration
        X_splits.append(split)
    return X_splits, eigen


def calculate_correlation_dissimilarity(sample_i, samples):
    distance_matrix = []
    for j, sample_j in enumerate(samples):
        s1 = np.sum(sample_i - sample_i[0])
        s2 = np.sum(sample_j - sample_j[0])
        if s1 < 1e-10 and s2 < 1e-10:  # check if constant
            corr = 1
        elif s1 < 1e-10 or s2 < 1e-10:
            corr = 0
        else:
            corr = np.corrcoef(sample_i, sample_j)[0, 1]
        d_corr = 1. - ((corr + 1) / 2.)
        distance_matrix.append(d_corr)
    return distance_matrix


def calculate_field_dissimilarity(sample_i, samples):
    distance_matrix = []
    for j, sample_j in enumerate(samples):
        maxi = np.amax(np.array([sample_i, sample_j]), axis=0)
        mini = np.amin(np.array([sample_i, sample_j]), axis=0)
        d_AB = 1. - np.sum(1 - maxi) / np.sum(1 - mini)
        if np.isinf(d_AB):
            d_AB = 0
        distance_matrix.append(d_AB)
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


def create_mds_result(output_path, distance_method, runs, durations, threads=1):
    Path(output_path).mkdir(parents=True, exist_ok=True)

    d_path = os.path.join(output_path, f'distance_{distance_method}.npy')
    if not os.path.exists(d_path):
        print('Calculate distance matrix ' + d_path)
        distance_matrix = calculate_distance_matrix(runs, distance_method, threads=threads)
        np.save(d_path, distance_matrix)
    else:
        distance_matrix = np.load(d_path)
    mds_path = os.path.join(output_path, f'mds_{distance_method}.pkl')
    mds_eigen_path = os.path.join(output_path, f'mds_eigen_{distance_method}.npy')
    if not os.path.exists(mds_path):
        X, eigen = calculate_mds(distance_matrix, durations)
        np.save(mds_eigen_path, eigen)
        with open(mds_path, 'wb') as handle:
            pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)


def plot_eigen(eigen_npy_path):
    eigen = np.load(eigen_npy_path)
    eigen /= np.sum(eigen)
    plt.bar(range(0, 10), eigen)
    plt.title('Eigenvalues of Phantom Dataset')
    plt.show()
