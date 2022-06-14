import argparse
import os
from termcolor import colored

from model.similarity_net import SimilarityNet
from utils import load_runs, batch, plot_runs

parser = argparse.ArgumentParser(description="Generate results from SimilarityNet")

parser.add_argument("-n", "--name", default='phantom_gradient', type=str)
parser.add_argument("-dp", "--data_path", default=os.path.join('data', 'phantom_gradient', 'samples', 'train'), type=str)
parser.add_argument("-cpp", "--checkpoint_path", default='checkpoints', type=str)

args = parser.parse_args()

checkpoint_path = os.path.join(args.checkpoint_path, args.name)

model = SimilarityNet.load_model(checkpoint_path)

result_runs, result_run_files, result_max_timesteps, result_durations = load_runs(args.data_path)

input_shape = (None, result_runs.shape[1], result_runs.shape[2])
model.build(input_shape=input_shape)

plot_runs(model, result_runs, result_durations)
