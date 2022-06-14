import os
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf


def save_json(path, data):
    with open(path, 'w') as fp:
        json.dump(data, fp, indent=4)


def load_json(path):
    with open(path, 'r') as fp:
        return json.load(fp)


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

    runs_np = np.array(runs)
    runs_np = np.reshape(runs_np, (runs_np.shape[0], runs_np.shape[1], np.prod(runs_np.shape[2:])))

    return runs_np, run_files, max_timesteps, durations


def batch(iterable, n=1):
    if n is None:
        return iter
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def plot_runs(model, all_runs, all_durations, latent_dim=None):
    runs = batch(all_runs, 1)
    latent_dim = model.args.latent_dim if latent_dim is None else latent_dim
    f, axs = plt.subplots(latent_dim, 1, sharex=True)
    latents = []
    for i, run in enumerate(runs):
        predictions, latent, decoder_attention_weights = model(run)
        latents.append(latent[0])
        for l in range(latent_dim):
            axs[l].plot(latent[0, :, l][:all_durations[i]])
            axs[l].set_ylabel(f'Latent feature {l}')
    axs[-1].set_xlabel('Time')

    plt.tight_layout(pad=0)
    plt.show()


def cosine_similarity(x, l=1.0):
    x0 = x[..., 0]
    x1 = x[..., 1]

    cos_sim = tf.keras.losses.cosine_similarity(x0, x1)
    return tf.math.abs(cos_sim)
