import argparse, os
import time

import tensorflow as tf

from termcolor import colored

from model.similarity_net import SimilarityNet
from train import Training
from utils import load_runs, plot_runs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Start training")

    parser.add_argument("-in", "--initializer", default=None, type=str)
    parser.add_argument("-a", "--activation", default='leaky_relu', type=str)
    parser.add_argument("-dr", "--dropout_rate", default=0., type=float)
    parser.add_argument("-dm", "--debug_model", default=False, type=bool)

    # Encoder
    parser.add_argument("-lin", "--latent_initializer", default=None, type=str)
    parser.add_argument("-la", "--latent_activation", default='leaky_relu', type=str)
    parser.add_argument('-els', '--encoder_layer_sizes', action='append', help='Encoder layer sizes')
    parser.add_argument("-ld", "--latent_dim", default=2, type=int)

    # Transformer
    parser.add_argument("-nh", "--num_heads", default=1, type=int)

    # Decoder
    parser.add_argument("-oin", "--output_initializer", default=None, type=str)
    parser.add_argument("-oa", "--output_activation", default='leaky_relu', type=str)
    parser.add_argument('-dls', '--decoder_layer_sizes', action='append', help='Decoder layer sizes')
    parser.add_argument("-od", "--output_dim", default=16384, type=int)

    # Training
    parser.add_argument("-n", "--name", default='phantom_final_large', type=str)
    parser.add_argument("-tdp", "--train_data_path", default=os.path.join('data', 'phantom_final', 'samples', 'train'), type=str)
    parser.add_argument("-vdp", "--val_data_path", default=None, type=str)
    parser.add_argument("-bs", "--batch_size", default=3, type=int)
    parser.add_argument("-e", "--epochs", default=500, type=int)
    parser.add_argument("-lrd", "--lr_decay", default=0.99, type=float)
    parser.add_argument("-lr", "--learning_rate", default=0.01, type=float)
    parser.add_argument("-sbv", '--save_best_val', default=False, type=bool)
    parser.add_argument("-l", "--log_dir", default='logs', type=str)
    parser.add_argument("-cpp", "--checkpoint_path", default='checkpoints', type=str)

    args = parser.parse_args()

    argsdict = vars(args)

    if args.name is None:
        args.name = f'{str(time.time())}-training'

    args.encoder_layer_sizes = [16, 8] if args.encoder_layer_sizes is None else args.encoder_layer_sizes
    args.decoder_layer_sizes = [16, 8] if args.decoder_layer_sizes is None else args.decoder_layer_sizes

    train_runs, train_run_files, train_max_timesteps, train_durations = load_runs(args.train_data_path)
    print(colored(f'Training {args.name} started on {len(train_runs)} runs.', 'green'))
    args.output_dim = train_runs.shape[-1]

    if args.val_data_path is not None:
        val_runs, val_run_files, val_max_timesteps, val_durations = load_runs(args.val_data_path)

    input_shape = (None, None, train_runs.shape[2])
    model = SimilarityNet(args)
    # model.build(input_shape=input_shape)

    args.loss_object = None
    args.optimizer = None

    trainer = Training(model, args)

    cb_runs, cb_run_files, cb_max_timesteps, cb_durations = load_runs(os.path.join('data', 'phantom_final', 'samples', 'train'))


    def callback(model, l, all_latents):
        plot_runs(model, cb_runs, cb_durations, latent_dim=2)


    trainer.train(train_runs, epoch_callback=callback)

    # dummy_input = tf.random.uniform(input_shape)
    # output = model(dummy_input)
    # print(output.shape)
    print('hi')
