import os
import time
from termcolor import colored

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils import batch


class Training:
    def __init__(self, model, args):
        self.model = model
        self.loss_object = args.loss_object if args.loss_object is not None else tf.keras.losses.MeanSquaredError()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='train_loss')
        self.optimizer = tf.keras.optimizers.Adam(args.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        # self.optimizer = args.optimizer if args.optimizer is not None else tf.keras.optimizers.Adam(args.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.save_best_val = args.save_best_val
        self.lr_decay = args.lr_decay

        self.train_summary_writer = None
        if args.log_dir is not None:
            self.train_summary_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, args.name))

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    @tf.function()
    def train_step(self, inp, trainable=True):
        with tf.GradientTape() as tape:
            model_outputs = self.model(inp)
            l = self.loss_function(inp, model_outputs[0])
            if trainable:
                gradients = tape.gradient(l, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                self.train_loss(l)
            else:
                self.val_loss(l)

        return model_outputs

    def train(self, train_runs, val_runs=None, epoch_callback=None):
        best_val_loss = 99999.
        tq = tqdm(range(self.epochs))
        for epoch in tq:
            start = time.time()
            self.train_loss.reset_states()
            self.val_loss.reset_states()

            batches = batch(train_runs, self.batch_size)

            best_loss = 99999.
            all_latents = []
            all_decoder_attention_weights = []

            for b in batches:
                predictions, latents, decoder_attention_weights = self.train_step(b, True)
                all_latents.append(latents)
                all_decoder_attention_weights.append(decoder_attention_weights)

            l = self.train_loss.result().numpy()
            tqdm_str = f'Loss: {round(l, 6)} LR: {round(self.optimizer.learning_rate.numpy(), 6)}'

            if not self.save_best_val and l < best_loss:
                best_loss = l
                self.model.save_model()
                tqdm_str = '[SAVED] ' + tqdm_str

            if val_runs is not None:
                val_batches = batch(val_runs, 1)
                v_losses = []
                v_all_enc_outputs = []
                v_all_decoder_attention_weights = []
                for vr in val_batches:
                    predictions, latents, decoder_attention_weights, = self.train_step(vr, False)
                    v_all_enc_outputs.append(latents)
                    v_all_decoder_attention_weights.append(decoder_attention_weights)
                    vl = self.val_loss.result()
                    v_losses.append(vl)
                vl = np.mean(v_losses)
                tqdm_str += f' Val. Loss: {round(vl, 6)}'

                if self.save_best_val and vl < best_val_loss:
                    best_val_loss = vl
                    self.model.save_model()
                    tqdm_str = '[SAVED] ' + tqdm_str

            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', l, step=epoch)
                if val_runs is not None:
                    tf.summary.scalar('val_loss', vl, step=epoch)

            tq.set_postfix_str(tqdm_str)
            self.optimizer.learning_rate.assign(self.optimizer.learning_rate.numpy() * self.lr_decay)
