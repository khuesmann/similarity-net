import argparse
import os
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import random_indices, load_netcfd4_variable, parse_params_from_run_name

SAMPLES_DIR = 'samples'


def create_run_samples(run, data_type, samples_path, data_path, file_type, simulation_dimension, field, max_timesteps=None, debug=False):
    timestep_files = os.listdir(os.path.join(data_path, run))
    timestep_files.sort()
    timestep_files = [tf for tf in timestep_files if not tf.endswith('.vvd')]
    run_sample_path = os.path.join(samples_path, data_type, run)
    if not os.path.exists(os.path.join(samples_path, data_type)):
        os.makedirs(os.path.join(samples_path, data_type))

    mini = 9999999
    maxi = -9999999

    timesteps = []
    for j, timestep_file in tqdm(enumerate(timestep_files)):
        full_ts_path = os.path.join(data_path, run, timestep_file)
        if file_type == 'binary':
            if simulation_dimension is None:
                print('For binary data, the "--simulation_dimension ( -sd ) [INT]" parameter has to be specified.')
                exit()
            timestep = np.fromfile(full_ts_path, np.float32, simulation_dimension * simulation_dimension * simulation_dimension)
        elif file_type == 'netcfd4' or file_type == 'netcfd':
            if field is None:
                print('For netcfd4 data, the "--netcfd_variable ( -nv ) [STR]" parameter has to be specified.')
                exit()
            timestep = load_netcfd4_variable(full_ts_path, field, None)
        elif file_type == 'npy':
            timestep = np.load(full_ts_path)
        elif file_type == 'hdf5':
            # print(full_ts_path)

            f = h5py.File(full_ts_path, 'r')
            fields = f.keys()
            channel = None
            for fi in fields:
                x = ET.fromstring(f[fi].attrs.get('metaData')).findall('metaData/MetaData/MetaItem')[2].find('value').attrib['value']
                if x == field:
                    channel = fi
                    break

            timestep = f[channel].value

        if timestep.shape[0] == 0:
            continue

        d = 128
        if len(timestep_files) > 1:
            mini = min(mini, timestep.min())
            maxi = max(maxi, timestep.max())
            random_samples = timestep.flatten()[random_indices(timestep.size, d * d)]
            random_samples = np.reshape(random_samples, (d, d))
            timesteps.append(random_samples)
        elif len(timestep_files) == 1:
            for t in timestep:
                mini = min(mini, t.min())
                maxi = max(maxi, t.max())
                random_samples = t.flatten()[random_indices(t.size, d * d)]
                random_samples = np.reshape(random_samples, (d, d))
                timesteps.append(random_samples)
        # plt.imshow(timesteps[-1])
        # plt.colorbar()
        # plt.show()
    np.save(os.path.join(run_sample_path + '.npy'), timesteps[:max_timesteps])
    return mini, maxi


def create_samples(data_path, ensemble_name, file_type, test_split, simulation_dimension, store_path, netcfd_variable, max_timesteps):
    print('Creating random-synthetic_small samples ...')
    runs = os.listdir(data_path)
    mini = 9999999
    maxi = -9999999

    samples_path = os.path.join(store_path, ensemble_name, SAMPLES_DIR)

    test_count = int(test_split * len(runs))
    if test_count != 0:
        test_indices = random_indices(len(runs), test_count)
        test_runs = np.array(runs)[test_indices]
        train_runs = np.delete(runs, test_indices)
    else:
        train_runs = runs
        test_result = []
    train_runs = [tr for tr in train_runs if not tr.startswith('.')]
    train_result = Parallel(n_jobs=9)(delayed(create_run_samples)(run, 'train', samples_path, data_path, file_type, simulation_dimension, netcfd_variable, max_timesteps) for run in tqdm(train_runs, desc=f'Train run'))
    if test_count != 0:
        test_result = Parallel(n_jobs=9)(delayed(create_run_samples)(run, 'test', samples_path, data_path, file_type, simulation_dimension, netcfd_variable, max_timesteps) for run in tqdm(test_runs, desc='test run'))

    extrema = np.array(train_result + test_result)
    min_max = (extrema[:, 0].min(), extrema[:, 1].max())
    np.save(os.path.join(store_path, ensemble_name, 'ensemble_extrema.npy'), min_max)

    normalize_samples(store_path, ensemble_name)
    print('... done')


def calculate_dissimilarity(sample_i, samples):
    distance_matrix = []
    for j, sample_j in enumerate(samples):
        maxi = np.amax(np.array([sample_i, sample_j]), axis=0)
        mini = np.amin(np.array([sample_i, sample_j]), axis=0)
        d_AB = 1 - np.sum(1 - maxi) / np.sum(1 - mini)
        distance_matrix.append(d_AB)
    return distance_matrix


def calculate_distance_matrix(store_path, ensemble_name, data_type='train', normalize_samples=True):
    samples = []
    sample_path = os.path.join(store_path, ensemble_name, SAMPLES_DIR, data_type)
    extrema = np.load(os.path.join(store_path, ensemble_name, 'ensemble_extrema.npy'))
    runs = os.listdir(sample_path)
    for run in runs:
        run_path = os.path.join(sample_path, run)
        timesteps = np.load(run_path)
        for timestep in timesteps:
            # if normalize_samples:
            #     timestep = (timestep - extrema[0]) / (extrema[1] - extrema[0])
            samples.append(timestep)

    samples = np.array(samples)

    distance_matrix = Parallel(n_jobs=4)(delayed(calculate_dissimilarity)(sample_i, samples) for sample_i in tqdm(samples, desc=f'{data_type} distance matrix'))
    distance_matrix = np.array(distance_matrix)

    np.save(os.path.join(store_path, ensemble_name, '{}_distance_matrix.npy'.format(data_type)), distance_matrix)
    print('Created distance matrix ({} x {})'.format(distance_matrix.shape[0], distance_matrix.shape[1]))


def calculate_mds(store_path, ensemble_name, data_type='train', params=None):
    print(f'Calculating MDS for {data_type}...')
    data_path = os.path.join(store_path, ensemble_name)
    distance_matrix = np.load(os.path.join(data_path, '{}_distance_matrix.npy'.format(data_type)))
    X, eigen = mds(distance_matrix, 10, 'cpu')

    X_split = []
    runs_path = os.path.join(data_path, SAMPLES_DIR, data_type)
    runs = os.listdir(runs_path)
    prev_i = 0
    for run in runs:
        timesteps = np.load(os.path.join(runs_path, run))
        timestep_count = len(timesteps)
        X_split.append([parse_params_from_run_name(ensemble_name, run), X[prev_i:prev_i + timestep_count]])
        prev_i += timestep_count

    np.save(os.path.join(data_path, '{}_X.npy'.format(data_type)), X)
    np.save(os.path.join(data_path, '{}_eigen.npy'.format(data_type)), eigen)
    np.save(os.path.join(data_path, '{}_X_split.npy'.format(data_type)), X_split)
    print('... done')


def plot(store_path, ensemble_name, dim=2, type='train'):
    data_path = os.path.join(store_path, ensemble_name)
    X = np.load(os.path.join(data_path, '{}_X_split.npy'.format(type)), allow_pickle=True)
    eigens = np.load(os.path.join(data_path, '{}_eigen.npy'.format(type)))
    # eigens = eigens / eigens.max()
    X = X.tolist()
    params = [x[0] for x in X]
    projs = [x[1] for x in X]
    mini = 99999
    maxi = -99999
    for i, p in enumerate(projs):
        p = p * eigens
        mini = min(mini, p.min())
        maxi = max(maxi, p.max())
        projs[i] = p
    for i, p in enumerate(projs):
        projs[i] = (projs[i] - mini) / (maxi - mini)

    # projs = (projs - projs.min()) / (projs.max() - projs.min())

    if dim == 1:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        for run in projs:
            ax.plot(run[:, 0] * eigens[0])

        plt.show()

        for run in projs:
            ax.plot(run[:, 0] * eigens[0])
        plt.show()

    if dim == 2:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        for run in projs:
            ax.scatter(run[:, 0] * eigens[0], run[:, 1] * eigens[1])
            ax.plot(run[:, 0] * eigens[0], run[:, 1] * eigens[1])

        plt.show()

        for run in projs:
            ax.scatter(run[:, 0] * eigens[0], run[:, 1] * eigens[1])
            ax.plot(run[:, 0] * eigens[0])
            ax.plot(run[:, 1] * eigens[1])
        plt.show()

    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for run in projs:
            ax.plot(xs=run[:, 0], ys=run[:, 1], zs=run[:, 2])

        plt.show()

    elif dim == 4:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

        for run in projs:
            ax.plot(run[:, 0], alpha=0.5)
            ax.plot(run[:, 1], alpha=0.5)
            ax.plot(run[:, 2], alpha=0.5)
            ax.plot(run[:, 3], alpha=0.5)
            plt.show()


def normalize_samples(store_path, ensemble_name):
    data_path = os.path.join(store_path, ensemble_name)
    train_data_path = os.path.join(data_path, 'samples', 'train')
    test_data_path = os.path.join(data_path, 'samples', 'test')
    extrema = np.load(os.path.join(data_path, 'ensemble_extrema.npy'))

    for train_file in os.listdir(train_data_path):
        full_path = os.path.join(train_data_path, train_file)
        train = np.load(full_path)
        train = (train - extrema[0]) / (extrema[1] - extrema[0])
        np.save(full_path, train)
    if os.path.exists(test_data_path):
        for test_file in os.listdir(test_data_path):
            full_path = os.path.join(test_data_path, test_file)
            train = np.load(full_path)
            train = (train - extrema[0]) / (extrema[1] - extrema[0])
            np.save(full_path, train)

    print('normalized samples')


def run(ensemble_path, ensemble_name, test_split, simulation_dimension, store_path, file_type, netcfd_variable, dims=3, max_timesteps=None):
    print(f"--- Calculating field similarity of {ensemble_name}---")
    start_time = time.time()

    dp = os.path.join(store_path, ensemble_name)
    create_samples(ensemble_path, ensemble_name, file_type=file_type, test_split=test_split, simulation_dimension=simulation_dimension, store_path=store_path, netcfd_variable=netcfd_variable, max_timesteps=max_timesteps)

    print("--- total: %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate field similarity (MDS, Eigens, Random Samples)")
    parser.add_argument("-ep", "--ensemble_path", required=True, type=str)
    parser.add_argument("-en", "--ensemble_name", required=True, type=str)
    parser.add_argument("-ts", "--test_split", default=0.2, type=float)
    parser.add_argument("-ft", "--file_type", default='binary', type=str)
    parser.add_argument("-sd", "--simulation_dimension", default=None, type=int)
    parser.add_argument("-nv", "--netcfd_variable", default=None, type=str)
    parser.add_argument("-sp", "--store_path", default='data', type=str)
    parser.add_argument("-d", "--dims", default=3, type=int)
    parser.add_argument("-mt", "--max_timesteps", default=None, type=int)

    args = parser.parse_args()

    print('Args: {}'.format(args))
    run(args.ensemble_path, args.ensemble_name, args.test_split, args.simulation_dimension, args.store_path, args.file_type, args.netcfd_variable, args.dims, args.max_timesteps)
