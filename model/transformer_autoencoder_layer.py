import argparse

import tensorflow as tf

layers = tf.keras.layers


class TransformerAutoencoderLayer(layers.Layer):
    def __init__(self, args):
        super(TransformerAutoencoderLayer, self).__init__()
        self.hidden_units = args.hidden_units
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(num_heads=args.num_heads, key_dim=args.latent_dim, dropout=args.dropout_rate)
        self.ff = layers.Dense(args.latent_dim, kernel_initializer=args.initializer, name='ff', activation=args.activation, activity_regularizer=tf.keras.regularizers.l1(10e-5) if args.is_sparse else None)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.fcs = []
        self.dropouts = []
        self.latent = None
        for units in args.hidden_units:
            self.fcs.append(layers.Dense(units, kernel_initializer=args.initializer, activation=args.activation, name='latent' if units is min(args.hidden_units) else None, activity_regularizer=tf.keras.regularizers.l1(10e-5) if args.is_sparse else None))
            self.dropouts.append(layers.Dropout(args.dropout_rate))
        self.add1 = layers.Add()
        self.add2 = layers.Add()

    def call(self, x, mask=None, training=True, **kwargs):
        out2 = x
        latent = None
        attention_weights = None
        for i, l in enumerate(self.fcs):
            if self.hidden_units[i] is min(self.hidden_units):
                latent = l(out2)
                out2 = l(out2)
                attention_output, attention_weights = self.mha(out2, out2, return_attention_scores=True, attention_mask=mask, training=training)
                out2 = self.add1([attention_output, out2])
                out2 = self.layer_norm2(out2)
                out3 = self.ff(out2)
                out2 = self.add2([out3, out2])

            else:
                out2 = l(out2)
                out2 = self.dropouts[i](out2, training=training)

        return out2, latent, attention_weights