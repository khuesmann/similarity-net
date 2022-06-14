import os

try:
    import netCDF4
except Exception as e:
    pass
import numpy as np


def batch(iterable, n=1):
    if n is None:
        return iter
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def get_run_names(runs_path):
    run_files = os.listdir(runs_path)
    run_files.sort()
    return run_files


def load_runs(runs_path, max_timesteps=None, max_runs=None):
    run_files = get_run_names(runs_path)
    runs = []
    durations = []
    max_ts = 0
    min_ts = 999999
    for run_file in run_files:
        run = np.load(os.path.join(runs_path, run_file))[:max_timesteps]
        mi, ma = run.min(axis=0), run.max(axis=0)
        durations.append(run.shape[0])
        max_ts = max(max_ts, run.shape[0])
        min_ts = min(min_ts, run.shape[0])
        if max_timesteps is not None and min_ts < max_timesteps:
            continue
        runs.append(np.expand_dims(np.expand_dims(run, -1), -1))
        if len(runs) == max_runs:
            break

    max_timesteps = max_ts

    if max_ts != min_ts:
        for i, run in enumerate(runs):
            runs[i] = np.pad(run, ((0, max_ts - run.shape[0]), (0, 0), (0, 0), (0, 0), (0, 0)), constant_values=-1)

    runs = np.array(runs)
    runs = np.reshape(runs, (runs.shape[0], runs.shape[1], np.prod(runs.shape[2:])))
    return runs, run_files, max_timesteps, durations


def load_netcfd4_variable(data_path, variable='DATA', transpose=[1, 2, 0]):
    f = netCDF4.Dataset(data_path)
    var = f.variables[variable]
    if transpose is not None:
        var = np.transpose(var, axes=transpose)
    return np.array(var)


def random_indices(total_size, indices_count, random_seed=126554):
    np.random.seed(random_seed)
    return np.random.choice(np.arange(total_size), indices_count, replace=False)


def get_distinguishable_color(i):
    colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ff0000', '#000000'
    ]
    return colors[i]


def parse_params_from_run_name(ensemble_name, run_dir):
    if 'semi_conductor' in ensemble_name:
        params = run_dir.split('_')
        probemut = float(params[1].split('*')[-1])
        excite_over_gap = float(params[4].split('=')[-1])
        pumpsigmax = float(params[5].split('=')[-1])
        pumparea = float(params[6].split('=')[-1])
        return [probemut, excite_over_gap, pumpsigmax, pumparea]
    elif 'white_dwarf' in ensemble_name:
        params = run_dir.split('wd')
        return [float(p) for p in params[1:]]
