import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from model.similarity_net import SimilarityNet
from utils import batch, load_runs


def loss_function(real, pred, enc_output=None):
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # TODO shoudn't it be -1 PAD TOKEN ?
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


@tf.function()
def train_step(inp, trainable=True):
    with tf.GradientTape() as tape:
        predictions, latents, attention_weights = transformer(inp)
        loss = loss_function(inp, predictions)

        if trainable:
            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
            train_loss(loss)
        else:
            val_loss(loss)

    return predictions, latents, attention_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Start training")
    # model attributes
    parser.add_argument("-nh", "--num_heads", default=1, type=int)
    parser.add_argument("-pd", "--projection_dim", default=8, type=int)
    parser.add_argument("-ld", "--latent_dim", default=2, type=int)
    parser.add_argument("-tlc", "--transformer_layer_count", default=1, type=int)
    parser.add_argument("-tdlc", "--transformer_dec_layer_count", default=1, type=int)
    parser.add_argument("-cp", "--checkpoint_path", default='checkpoint', type=str)
    parser.add_argument("-in", "--initializer", default=None, type=str)
    parser.add_argument("-a", "--activation", default='leaky_relu', type=str)
    parser.add_argument("-oa", "--out_activation", default='leaky_relu', type=str)

    # training attributes
    parser.add_argument("-dr", "--dropout_rate", default=0., type=float)
    parser.add_argument("-tdp", "--train_data_path", required=True, type=str)
    parser.add_argument("-vdp", "--val_data_path", default=None, type=str)
    parser.add_argument("-n", "--name", required=True, type=str)
    parser.add_argument("-lr", "--learning_rate", default=0.01, type=float)
    parser.add_argument("-lrd", "--lr_decay", default=0.999, type=float)
    parser.add_argument("-e", "--epochs", default=500, type=int)
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("-pf", "--plot_freq", default=100, type=int)
    parser.add_argument("-log", "--log_dir", default='logs', type=str)
    parser.add_argument("-sbv", "--save_best_val", default=False, type=bool)
    parser.add_argument("-is", "--is_sparse", default=False, type=bool)

    args = parser.parse_args()

    train_summary_writer = None
    if args.log_dir is not None:
        train_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, args.name))

    # dataset related
    train_runs, train_run_files, train_max_timesteps, train_durations = load_runs(args.train_data_path)
    if args.val_data_path is not None:
        val_runs, val_run_files, val_max_timesteps, val_durations = load_runs(args.val_data_path)
    args.num_timesteps = train_max_timesteps if args.val_data_path is None else max(train_max_timesteps, val_max_timesteps)
    args.output_size = train_runs.shape[-1]

    Path(os.path.join(args.checkpoint_path, args.name)).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(os.path.join(args.checkpoint_path, args.name), 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # create the transformer
    transformer = SimilarityNet.make(args)

    # loss_object = tf.keras.losses.BinaryCrossentropy()
    loss_object = tf.keras.losses.MeanSquaredError()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='train_loss')

    optimizer = tf.keras.optimizers.Adam(args.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model_path = os.path.join(os.path.join(args.checkpoint_path, args.name), 'model.ckpt')
    transformer.checkpoint_manager(model_path, optimizer)

    tq = tqdm(range(args.epochs))
    best_val_loss = 99999.

    for epoch in tq:
        start = time.time()
        train_loss.reset_states()
        val_loss.reset_states()

        batches = batch(train_runs, args.batch_size)

        best_loss = 99999.
        all_latents = []
        all_decoder_attention_weights = []
        for b in batches:
            mask = np.ma.make_mask(b + 1)
            predictions, latents, decoder_attention_weights = train_step(b, True)
            all_latents.append(latents)
            all_decoder_attention_weights.append(decoder_attention_weights)

        l = train_loss.result().numpy()
        if not args.save_best_val and l < best_loss:
            best_loss = l
            transformer.save_model()

        if args.val_data_path is not None:
            val_batches = batch(val_runs, 1)
            v_losses = []
            v_all_enc_outputs = []
            v_all_decoder_attention_weights = []
            for vr in val_batches:
                predictions, latents, decoder_attention_weights, = train_step(vr, False)
                v_all_enc_outputs.append(latents)
                v_all_decoder_attention_weights.append(decoder_attention_weights)
                vl = val_loss.result()
                v_losses.append(vl)
            vl = np.mean(v_losses)
            if args.save_best_val and vl < best_val_loss:
                best_val_loss = vl
                transformer.save_model()
                print('saved')
            # else:
            #     transformer.checkpoint_manager(model_path, optimizer)

        tqdm_str = f'Loss: {round(l, 6)} LR: {round(optimizer.learning_rate.numpy(), 6)}'

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', l, step=epoch)
            if args.val_data_path is not None:
                tf.summary.scalar('val_loss', vl, step=epoch)
                tqdm_str += f' Val. Loss: {round(vl, 6)}'
        tq.set_postfix_str(tqdm_str)

        # transformer.checkpoint_manager(model_path, optimizer, silent=True)
        optimizer.learning_rate.assign(optimizer.learning_rate.numpy() * args.lr_decay)
    print(f'Best Loss: {best_loss} Best Val Loss: {best_val_loss}')
