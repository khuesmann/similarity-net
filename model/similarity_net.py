import tensorflow as tf

from model.transformer_autoencoder_layer import TransformerAutoencoderLayer


layers = tf.keras.layers


class SimilarityNet(tf.keras.Model):
    def __init__(self, args):
        super(SimilarityNet, self).__init__()
        self.ckpt_manager = None
        self.projection_in = layers.Dense(args.projection_dim, kernel_initializer=args.initializer, activation=args.activation)
        self.encoded = []
        for i in range(args.transformer_layer_count):
            self.encoded.append(TransformerAutoencoderLayer(args))

        self.projection_out = layers.Dense(args.output_size, kernel_initializer=args.initializer, activation=args.out_activation)

    def call(self, x, mask=None, training=True, **kwargs):
        x = self.projection_in(x)
        latents = []
        attentions = []
        for encode in self.encoded:
            x, latent, attention = encode(x, mask, training)
            latents.append(latent)
            attentions.append(attention)
            
        x = self.projection_out(x, training=training)

        return x, latents, attentions

    def checkpoint_manager(self, checkpoint_path, optimizer=None, silent=False):
        ckpt = tf.train.Checkpoint(selfattention=self)
        self.ckpt_manager = tf.train.CheckpointManager(
            tf.train.Checkpoint(selfattention=self),
            checkpoint_path,
            max_to_keep=2
        )
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
            if not silent:
                print('Latest checkpoint restored!!')

    def save_model(self):
        self.ckpt_manager.save()

    @staticmethod
    def make(args, example_trans_out=None):
        args.activation = getattr(tf.nn, args.activation)
        args.out_activation = getattr(tf.nn, args.out_activation)
        args.hidden_units = [args.projection_dim // 2, args.latent_dim, args.projection_dim // 2]
        args.is_sparse = False

        transformer = SimilarityNet(args)
        if example_trans_out is None:
            transformer(tf.random.uniform((13, 256, 128 * 128)))
        else:
            transformer(example_trans_out)

        transformer.summary()

        return transformer