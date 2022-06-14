import argparse
import os
from pathlib import Path

from termcolor import colored

import tensorflow as tf

from utils import save_json, load_json, Struct, cosine_similarity

layers = tf.keras.layers


class Encoder(tf.keras.Model):
    def __init__(self, args: argparse.Namespace):
        super(Encoder, self).__init__()

        self.fcs = []
        self.dropouts = []
        for layer_size in args.encoder_layer_sizes:
            self.fcs.append(layers.Dense(int(layer_size), kernel_initializer=args.initializer, activation=args.activation))
            self.dropouts.append(layers.Dropout(args.dropout_rate))

        # self.latent = layers.Dense(args.latent_dim, kernel_initializer=args.latent_initializer, activation=cosine_similarity)
        self.latent = layers.Dense(args.latent_dim, kernel_initializer=args.latent_initializer)
        self.debug_model = args.debug_model

    def call(self, x):
        out = x
        for fc in self.fcs:
            out = fc(out)
            if self.debug_model:
                print('Decoder:', out.shape)
        out = self.latent(out)
        if self.debug_model:
            print('Latent:', out.shape)
        return out


class TransformerLayer(tf.keras.Model):
    def __init__(self, args: argparse.Namespace):
        super(TransformerLayer, self).__init__()
        self.mha = layers.MultiHeadAttention(num_heads=args.num_heads, key_dim=args.latent_dim, dropout=args.dropout_rate)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ff = layers.Dense(args.latent_dim, kernel_initializer=args.initializer, name='ff', activation=args.activation, activity_regularizer=tf.keras.regularizers.l1(10e-5))
        self.add1 = layers.Add()
        self.add2 = layers.Add()
        self.debug_model = args.debug_model

    def call(self, x, mask=None, training=True):
        out = x
        attention_output, attention_weights = self.mha(out, out, return_attention_scores=True, attention_mask=mask, training=training)
        if self.debug_model:
            print('Attention output:', attention_output.shape)
            print('Attention weights:', attention_weights.shape)
        out = self.add1([attention_output, out])
        out = self.layer_norm2(out)
        if self.debug_model:
            print('Transformer Out 1:', out.shape)
        out2 = self.ff(out)
        if self.debug_model:
            print('Transformer Out 2:', out2.shape)
        out = self.add2([out2, out])

        if self.debug_model:
            print('Transformer Output:', out.shape)

        return out, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, args: argparse.Namespace):
        super(Decoder, self).__init__()

        self.fcs = []
        self.dropouts = []
        for layer_size in args.decoder_layer_sizes:
            self.fcs.append(layers.Dense(layer_size, kernel_initializer=args.initializer, activation=args.activation))
            self.dropouts.append(layers.Dropout(args.dropout_rate))

        self.out = layers.Dense(args.output_dim, kernel_initializer=args.output_initializer, activation=args.output_activation)
        self.debug_model = args.debug_model

    def call(self, x):
        out = x
        for fc in self.fcs:
            out = fc(out)
            if self.debug_model:
                print('Decoder:', out.shape)
        out = self.out(out)
        if self.debug_model:
            print('Out:', out.shape)
        return out


class SimilarityNet(tf.keras.Model):
    def __init__(self, args: argparse.Namespace):
        super(SimilarityNet, self).__init__()

        self.args = args
        self.model_path = os.path.join(args.checkpoint_path, args.name)

        self.encoder = Encoder(args)
        self.transformer_layer = TransformerLayer(args)
        self.decoder = Decoder(args)

    def call(self, x):
        latent = self.encoder(x)
        transformer_out, attention_weights = self.transformer_layer(latent)
        out = self.decoder(transformer_out)

        return out, latent, attention_weights

    def print_summary(self):
        self.summary()
        print('#' * 100)
        self.encoder.summary()
        print('#' * 100)
        self.transformer_layer.summary()
        print('#' * 100)
        self.decoder.summary()
        print('#' * 100)

    def save_model(self):
        Path(self.model_path).mkdir(parents=True, exist_ok=True)

        self.save_weights(os.path.join(self.model_path, 'weights.h5'))
        save_json(os.path.join(self.model_path, 'args.json'), vars(self.args))

    @staticmethod
    def load_model(path):
        args = load_json(os.path.join(path, 'args.json'))
        a = Struct(**args)
        model = SimilarityNet(a)
        model(tf.random.uniform((13, 256, 128 * 128)))
        model.load_weights(os.path.join(path, 'weights.h5'))
        return model
