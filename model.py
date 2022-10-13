import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.layers as layers
import numpy as np
from tensorflow.keras.regularizers import L1, L2
from tqdm.notebook import tqdm
from tensorflow.keras.layers import Layer


physical_gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_gpus[0], True)
tf.config.optimizer.set_jit(True)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['XLA_FAGS'] = '--xla_gpu_cuda_data_dir=/path/to/cuda'


class Encoder(Model):
    def __init__(self, kernel, strides, out_length, out_channel=None):
        super(Encoder, self).__init__()

        self.channel = None
        if out_channel is not None:
            self.channel = Sequential()
            self.channel.add(layers.Conv1D(kernel_size=kernel, filters=out_channel, strides=strides,
                                           activation=tf.nn.leaky_relu, kernel_regularizer=L2(1e-3)))
            self.channel.add(layers.Conv1D(kernel_size=kernel, filters=out_channel, padding='same',
                                           activation=tf.nn.leaky_relu, kernel_regularizer=L2(1e-3)))

        self.dense = Sequential()
        self.dense.add(layers.Dense(out_length, activation=tf.nn.leaky_relu,
                                    kernel_regularizer=L2(1e-3)))
        self.dense.add(layers.Dense(out_length, activation=tf.nn.leaky_relu,
                                    kernel_regularizer=L2(1e-3)))
        self.compress_time = layers.TimeDistributed(self.dense)

        self.norm = layers.BatchNormalization()

    def call(self, x, trainable=True):
        logits = self.norm(x, training=trainable)
        if self.channel is None:
            return tf.transpose(self.compress_time(tf.transpose(logits, [0, 2, 1])), [0, 2, 1])
        else:
            return tf.transpose(self.compress_time(tf.transpose(self.channel(logits), [0, 2, 1])), [0, 2, 1])


class Decoder(Model):
    def __init__(self, time_out, channel_out=None):
        super(Decoder, self).__init__()
        self.dense = Sequential()
        self.dense.add(layers.Dense(time_out, activation=tf.nn.leaky_relu,
                                    kernel_regularizer=L2(1e-3)))
        self.dense.add(layers.Dense(time_out, activation=tf.nn.leaky_relu,
                                    kernel_regularizer=L2(1e-3)))
        self.decompress_time = layers.TimeDistributed(self.dense)

        self.decompress_channel = None
        if channel_out is not None:
            self.decompress_channel = Sequential()
            self.decompress_channel.add(layers.Conv1D(filters=channel_out, kernel_size=1,
                                                      activation=tf.nn.leaky_relu))
            self.decompress_channel.add(layers.Conv1D(filters=channel_out, kernel_size=1,
                                                      activation=tf.nn.leaky_relu))

    def call(self, x, trainable=True):
        if self.decompress_channel is not None:
            logits = tf.transpose(self.decompress_channel(x), [0, 2, 1])
        else:
            logits = tf.transpose(x, [0, 2, 1])
        return tf.transpose(self.decompress_time(logits), [0, 2, 1])


class Generator(Model):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.encoder, self.decoder = Sequential(), Sequential()
        param_en, param_de = params[0], params[1]
        for param in param_en:
            self.encoder.add(Encoder(*param))
        for param in param_de:
            self.decoder.add(Decoder(*param))

    @tf.function
    def call(self, x, trainable=True):
        return self.decoder(self.encoder(x, trainable), trainable)

    @tf.function
    def en_call(self, x, trainable=True):
        return self.encoder(x, trainable)

    @tf.function
    def de_call(self, x, trainable=True):
        return self.decoder(x, trainable)


class D_layer(Layer):
    def __init__(self, kernel_size, strides, filters):
        super(D_layer, self).__init__()
        self.conv_1 = layers.Conv1D(filters, kernel_size, strides,
                                    activation=tf.nn.leaky_relu)
        self.conv_2 = layers.Conv1D(filters, kernel_size, strides,
                                    activation=tf.nn.leaky_relu, padding='same')
        self.norm = layers.BatchNormalization()

    def call(self, x, trainable=True):
        return self.norm(self.conv_2(self.conv_1(x)), training=trainable)


class Discriminator(Model):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.model = Sequential()
        for param in params:
            self.model.add(D_layer(*param))

    @tf.function
    def call(self, x, trainable=True):
        return self.model(x, trainable)


class Tan(Model):
    def __init__(self, gen_params, dis_params, epochs, k, line=10, std=1, factor=0.02,
                 lr=0.01, d_lr=0.01, r_lr=0.005, l_lr=0.02):
        super(Tan, self).__init__()
        self.gen = Generator(gen_params)
        self.disc = Discriminator(dis_params)

        self.optimizer = tf.optimizers.Adam(lr)
        self.d_optimizer = tf.optimizers.Adam(d_lr)
        self.r_optimizer = tf.optimizers.Adam(r_lr)
        self.l_optimizer = tf.optimizers.Adam(l_lr)

        self.k = k
        self.line = line
        self.std = std
        self.factor = factor
        self.dis_epochs, self.gen_epochs = epochs[0], epochs[1]

        self.loss = []
        self.dist = []
        self.d_loss = []

    @tf.function
    def drop_loss(self, raw, recon, k, stochastic=False):
        loss = tf.transpose(tf.reduce_mean(tf.abs(raw - recon), axis=1))  # loss: (feature, batch)
        if stochastic:
            max_loss = tf.reduce_max(loss, axis=1, keepdims=True)  # (feature, 1)
            p = tf.nn.softmax(tf.multiply(loss, 1 / max_loss), axis=1)  # (feature, batch)

            # Algorithm A ，《Weighted Random Sampling》
            values = tf.math.pow(tf.random.uniform(loss.shape), 1 / p)  # (feature, batch)
            __, indices = tf.math.top_k(values, k=k)
        else:
            __, indices = tf.math.top_k(loss, k=k)
        dropped_loss = tf.reduce_sum(tf.gather(loss, indices, batch_dims=1), axis=1)
        size = tf.cast((loss.shape[1] - k), tf.float32)
        return (tf.reduce_sum(loss, axis=1, keepdims=False) - dropped_loss) / size

    @staticmethod
    def choose(k):
        def sum_(x):
            loss = x[:-k]
            indices = tf.cast(x[-k:], tf.int32)
            return tf.reduce_sum(tf.gather(loss, indices))

        return sum_

    @tf.function
    def latent_generator(self, latent):

        def l_generator(latent_, std=self.std):
            delta = latent_[1:, :] - latent_[:-1, :]
            logits = latent_[:-1, :] + tf.multiply(delta, tf.random.normal(delta.shape, stddev=std))
            return logits

        logits_1 = tf.map_fn(l_generator, tf.transpose(latent, [2, 0, 1]))
        return tf.transpose(logits_1, [1, 2, 0])

    def train(self, train_db_):
        for i in tqdm(range(len(self.dis_epochs))):
            for epoch in tf.range(self.dis_epochs[i]):
                for x in train_db_:
                    with tf.GradientTape() as dis_tape:
                        loss_1 = -tf.reduce_mean(tf.reduce_mean(tf.abs(self.disc(self.gen(x)) - self.disc(x)), axis=1),
                                                 axis=0)
                    dis_gradients = dis_tape.gradient(loss_1, self.disc.trainable_variables)
                    self.d_optimizer.apply_gradients(zip(dis_gradients, self.disc.trainable_variables))
                self.d_loss.append(tf.reduce_mean(loss_1))

                for x in train_db_:
                    with tf.GradientTape() as r:
                        latent = self.gen.en_call(x, trainable=True)

                        samples = self.latent_generator(latent)
                        out_r = self.gen.de_call(samples, trainable=True)
                        mae_r = tf.reduce_mean(tf.reduce_mean(tf.abs(out_r - self.line), axis=1), axis=0)
                    r_gradients = r.gradient(mae_r, self.gen.decoder.trainable_variables)
                    self.r_optimizer.apply_gradients(zip(r_gradients, self.gen.decoder.trainable_variables))
                self.dist.append(tf.reduce_mean(mae_r))

            for epoch in tf.range(self.gen_epochs[i]):
                for x in train_db_:
                    with tf.GradientTape() as joint_tape:
                        out_j = self.gen(x, trainable=True)
                        mae_2 = self.drop_loss(x, out_j, self.k)
                        dis_ = self.drop_loss(self.disc(x), self.disc(out_j), self.k)
                        joint_loss = mae_2 + self.factor * dis_
                    joint_gradients = joint_tape.gradient(joint_loss, self.gen.trainable_variables)
                    self.optimizer.apply_gradients(zip(joint_gradients, self.gen.trainable_variables))
                self.loss.append(tf.reduce_mean(mae_2))

    def get_loss(self):
        return self.loss, self.dist, self.d_loss

