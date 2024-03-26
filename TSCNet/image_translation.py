# import tensorflow as tf

from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Conv2D, Dropout, Activation, Dense, BatchNormalization, Lambda, UpSampling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape
from tensorflow.keras.activations import relu, sigmoid, tanh
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as Kb
from attention import PAM
from keras.layers.merge import concatenate
from keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
import tensorflow as tf
import matplotlib.pyplot as plt

class Wavelet(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def WaveletTransformAxisY(self, batch_img):
        odd_img = batch_img[:, 0::2]
        even_img = batch_img[:, 1::2]
        if odd_img.shape == even_img.shape:
            L = (odd_img + even_img) / 2.0
            H = (odd_img - even_img) / 2.0

        else:
            extra = tf.zeros([batch_img.shape[0], 1, batch_img.shape[2]], tf.float32)
            even_img = tf.concat([even_img, extra], 1)
            L = (odd_img + even_img) / 2.0
            H = (odd_img - even_img) / 2.0

        return L, H

    def WaveletTransformAxisX(self, batch_img):
        tmp_batch = Kb.permute_dimensions(batch_img, [0, 2, 1])[:, :, ::-1]
        _dst_L, _dst_H = self.WaveletTransformAxisY(tmp_batch)
        dst_L = Kb.permute_dimensions(_dst_L, [0, 2, 1])[:, ::-1, ...]
        dst_H = Kb.permute_dimensions(_dst_H, [0, 2, 1])[:, ::-1, ...]
        return dst_L, dst_H

    def call(self, batch_image):
        wavelet_data_l1 = []

        batch_image = Kb.permute_dimensions(batch_image, [0, 3, 1, 2])
        channel = batch_image.shape[1]
        for i in range(channel):
            wavelet_L, wavelet_H = self.WaveletTransformAxisY(batch_image[:, i])
            wavelet_LL, wavelet_LH = self.WaveletTransformAxisX(wavelet_L)
            wavelet_HL, wavelet_HH = self.WaveletTransformAxisX(wavelet_H)

            wavelet_data_l1.extend([wavelet_LL, wavelet_LH, wavelet_HL, wavelet_HH])

        transform_batch_l2 = Kb.stack(wavelet_data_l1, axis=1)
        decom_level_2 = Kb.permute_dimensions(transform_batch_l2, [0, 2, 3, 1])

        return decom_level_2

class Rewavelet(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def ReconstructionX(self, batch_img1, batch_img2):
        temp_batch_img1 = Kb.permute_dimensions(batch_img1, [0, 2, 1])
        temp_batch_img2 = Kb.permute_dimensions(batch_img2, [0, 2, 1])

        re_wavelet_ = self.ReconstructionY(temp_batch_img1, temp_batch_img2)
        re_wavelet_ = Kb.permute_dimensions(re_wavelet_, [0, 2, 1])

        return re_wavelet_

    def ReconstructionY(self, batch_img1, batch_img2):
        re_wavelet_odd = batch_img1 + batch_img2
        re_wavelet_even = batch_img1 - batch_img2

        temp = []
        for i in range(0, re_wavelet_odd.shape[1]):
            temp.append(re_wavelet_odd[:, i])
            temp.append(re_wavelet_even[:, i])

        if batch_img1.shape[1] == 384:
            del temp[-1]

        if batch_img1.shape[1] == 438:
            del temp[-1]

        if batch_img1.shape[1] == 461:
            del temp[-1]

        if batch_img1.shape[1] == 297:
            del temp[-1]

        if batch_img1.shape[1] == 252:
            del temp[-1]

        if batch_img1.shape[1] == 248:
            del temp[-1]

        re_wavelet_ = tf.stack(temp, axis=1)

        return re_wavelet_

    def call(self, batch_image):
        re_data = []
        batch_image = Kb.permute_dimensions(batch_image, [0, 3, 1, 2])
        channel = batch_image.shape[1]

        for i in range(0, channel, 4):
            L = self.ReconstructionX(batch_image[:, i], batch_image[:, i + 1])
            H = self.ReconstructionX(batch_image[:, i + 2], batch_image[:, i + 3])
            re = self.ReconstructionY(L, H)

            re_data.append(re)

        re_batch = Kb.stack(re_data, axis=1)
        re_image = Kb.permute_dimensions(re_batch, [0, 2, 3, 1])

        return re_image

def regularized_padded_conv(*args, **kwargs):
    return Conv2D(*args, **kwargs, padding='same', use_bias=False, kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(5e-4))

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, in_planes, ratio=12):
        super(ChannelAttention, self).__init__()
        self.avg = GlobalAveragePooling2D()
        self.max = GlobalMaxPooling2D()
        self.conv1 = Conv2D(in_planes // ratio, kernel_size=1, strides=1, padding='same',
                                   kernel_regularizer=regularizers.l2(5e-4),
                                   use_bias=True, activation=tf.nn.relu)
        self.conv2 = Conv2D(in_planes, kernel_size=1, strides=1, padding='same',
                                   kernel_regularizer=regularizers.l2(5e-4),
                                   use_bias=True)

    def call(self, inputs):
        avg = self.avg(inputs)
        max = self.max(inputs)
        avg = Reshape((1, 1, avg.shape[1]))(avg)
        max = Reshape((1, 1, max.shape[1]))(max)
        avg_out = self.conv2(self.conv1(avg))
        max_out = self.conv2(self.conv1(max))
        out = avg_out + max_out
        out = tf.nn.sigmoid(out)

        return out

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = regularized_padded_conv(1, kernel_size=kernel_size, strides=1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3)
        max_out = tf.reduce_max(inputs, axis=3)
        out = tf.stack([avg_out, max_out], axis=3)
        out = self.conv1(out)

        return out

class ImageTranslationNetwork(Model):
    def __init__(
        self,
        input_chs,
        filter_spec,
        name,
        l2_lambda=1e-3,
        leaky_alpha=0.3,
        dropout_rate=0.2,
        dtype="float32",
    ):
        """
            Inputs:
                input_chs -         int, number of channels in input
                filter_spec -       list of integers specifying the filtercount
                                    for the respective layers
                name -              str, name of model
                leaky_alpha=0.3 -   float in [0,1], passed to the RELU
                                    activation of all but the last layer
                dropout_rate=0.2 -  float in [0,1], specifying the dropout
                                    probability in training time for all but
                                    the last layer
                dtype='float64' -   str or dtype, datatype of model
            Outputs:
                None
        """
        super().__init__(name=name, dtype=dtype)

        self.leaky_alpha = leaky_alpha
        self.dropout = Dropout(dropout_rate, dtype=dtype)

        conv_specs = {
            "kernel_size": 3,
            "strides": 1,
            "kernel_initializer": "GlorotNormal",
            "padding": "same",
            "kernel_regularizer": l2(l2_lambda),
            # "bias_regularizer": l2(l2_lambda),
            "dtype": dtype,
        }

        self.layers_ = []
        if self.name == 'enc_X' or self.name == 'enc_Y':
            wavelet = Wavelet()
            self.layers_.append(wavelet)

            layer = Conv2D(
                filter_spec[0],
                input_shape=(None, None, input_chs * 4),
                name=f"{name}-{0:02d}",
                **conv_specs,
            )
            self.layers_.append(layer)

            for l, n_filters in enumerate(filter_spec[1:]):
                layer = Conv2D(n_filters, name=f"{name}-{l + 1:02d}", **conv_specs)
                self.layers_.append(layer)
                if l == 0:
                    cam = ChannelAttention(50)
                    self.layers_.append(cam)
                    pam = SpatialAttention()
                    self.layers_.append(pam)

        if self.name == 'dec_X' or self.name == 'dec_Y':
            layer = Conv2D(
                filter_spec[0],
                input_shape=(None, None, input_chs),
                name=f"{name}-{0:02d}",
                **conv_specs,
            )
            self.layers_.append(layer)
            cam2 = ChannelAttention(50)
            self.layers_.append(cam2)
            pam2 = SpatialAttention()
            self.layers_.append(pam2)

            for l, n_filters in enumerate(filter_spec[1:]):
                layer = Conv2D(n_filters, name=f"{name}-{l + 1:02d}", **conv_specs)
                self.layers_.append(layer)

            rewavelet = Rewavelet()
            self.layers_.append(rewavelet)


    def call(self, inputs, training=False):
        """ Implements the feed forward part of the network """
        x = inputs
        for layer in self.layers_[:-1]:
            x = layer(x)
            if layer.name == 'channel_attention' or layer.name == 'channel_attention_1' or layer.name == 'channel_attention_2' or layer.name == 'channel_attention_3':
                x = x * x1
            if layer.name == 'spatial_attention' or layer.name == 'spatial_attention_1' or layer.name == 'spatial_attention_2' or layer.name == 'spatial_attention_3':
                x = x * x2
            x = relu(x, alpha=self.leaky_alpha)
            x = self.dropout(x, training)
            if layer.name == 'enc_X-01' or layer.name == 'enc_Y-01' or layer.name == 'dec_X-00' or layer.name == 'dec_Y-00':
                x1 = x
            if layer.name == 'channel_attention' or layer.name == 'channel_attention_1' or layer.name == 'channel_attention_2' or layer.name == 'channel_attention_3':
                x2 = x
        x = self.layers_[-1](x)
        return tanh(x)

class Discriminator(Model):
    """
        CGAN by .. et. al discriminator
    """

    def __init__(
        self,
        shapes,
        filter_spec,
        name,
        l2_lambda=1e-3,
        leaky_alpha=0.3,
        dropout_rate=0.2,
        dtype="float32",
    ):
        """
            Inputs:
                input_chs -         int, number of channels in input
                filter_spec -       list of integers specifying the filtercount
                                    for the respective layers
                name -              str, name of model
                leaky_alpha=0.3 -   float in [0,1], passed to the RELU
                                    activation of all but the last layer
                dropout_rate=0.2 -  float in [0,1], specifying the dropout
                                    probability in training time for all but
                                    the last layer
                dtype='float64' -   str or dtype, datatype of model
            Outputs:
                None
        """
        super().__init__(name=name, dtype=dtype)
        self.leaky_alpha = leaky_alpha
        self.dropout = Dropout(dropout_rate, dtype=dtype)
        conv_specs = {
            "kernel_initializer": "GlorotNormal",
            # "kernel_regularizer": l2(l2_lambda),
            "dtype": dtype,
        }
        layer = Dense(
            filter_spec[0],
            input_shape=(None, shapes[0], shapes[0], shapes[1]),
            name=f"{name}-{0:02d}",
            **conv_specs,
        )
        self.layers_ = [layer]
        for l, n_filters in enumerate(filter_spec[1:]):
            layer = Dense(n_filters, name=f"{name}-{l+1:02d}", **conv_specs)
            self.layers_.append(layer)

    def call(self, inputs, training=False):
        """ Implements the feed forward part of the network """
        x = inputs
        for layer in self.layers_[:-1]:
            x = layer(x)
            x = relu(x, alpha=self.leaky_alpha)
            x = self.dropout(x, training)
        x = self.layers_[-1](x)
        return sigmoid(x)


class Generator(Model):
    """
        CGAN by .. et. al Generator and Approximator
    """

    def __init__(
        self,
        shapes,
        filter_spec,
        name,
        l2_lambda=1e-3,
        leaky_alpha=0.3,
        dropout_rate=0.2,
        dtype="float32",
    ):
        """
            Inputs:
                input_chs -         int, number of channels in input
                filter_spec -       list of integers specifying the filtercount
                                    for the respective layers
                name -              str, name of model
                leaky_alpha=0.3 -   float in [0,1], passed to the RELU
                                    activation of all but the last layer
                dropout_rate=0.2 -  float in [0,1], specifying the dropout
                                    probability in training time for all but
                                    the last layer
                dtype='float64' -   str or dtype, datatype of model
            Outputs:
                None
        """
        super().__init__(name=name, dtype=dtype)
        self.leaky_alpha = leaky_alpha
        self.dropout = Dropout(dropout_rate, dtype=dtype)
        self.ps = shapes[0]
        self.shape_out = filter_spec[-1]
        conv_specs = {
            "kernel_initializer": "GlorotNormal",
            # "kernel_regularizer": l2(l2_lambda),
            "dtype": dtype,
        }
        layer = Dense(
            filter_spec[0],
            input_shape=(None, self.ps, self.ps, shapes[1]),
            name=f"{name}-{0:02d}",
            **conv_specs,
        )
        self.layers_ = [layer]
        for l, n_filters in enumerate(filter_spec[1:]):
            layer = Dense(n_filters, name=f"{name}-{l+1:02d}", **conv_specs)
            self.layers_.append(layer)

    def call(self, inputs, training=False):
        """ Implements the feed forward part of the network """
        x = inputs
        for layer in self.layers_[:-1]:
            x = layer(x)
            x = relu(x, alpha=self.leaky_alpha)
            x = self.dropout(x, training)
        x = self.layers_[-1](x)
        x = relu(x, alpha=self.leaky_alpha)
        return tf.reshape(x, [-1, self.ps, self.ps, self.shape_out])


class CouplingNetwork(Model):
    """
        Same as network in Luigis cycle_prior.

        Not supporting discriminator / Fully connected output.
        Support for this should be implemented as a separate class.
    """

    def __init__(
        self,
        input_chs,
        filter_spec,
        name,
        decoder=False,
        l2_lambda=1e-3,
        dtype="float32",
    ):
        """
            Inputs:
                input_chs -         int, number of channels in input
                filter_spec -       list of integers specifying the filtercount
                                    for the respective layers
                name -              str, name of model
                leaky_alpha=0.3 -   float in [0,1], passed to the RELU
                                    activation of all but the last layer
                dropout_rate=0.2 -  float in [0,1], specifying the dropout
                                    probability in training time for all but
                                    the last layer
                dtype='float64' -   str or dtype, datatype of model
            Outputs:
                None
        """
        super().__init__(name=name, dtype=dtype)
        self.decoder = decoder
        conv_specs = {
            "kernel_size": 3,
            "strides": 1,
            "kernel_initializer": "GlorotNormal",
            "padding": "same",
            "kernel_regularizer": l2(l2_lambda),
            # "bias_regularizer": l2(l2_lambda),
            "dtype": dtype,
        }
        layer = Conv2D(
            filter_spec[0],
            input_shape=(None, None, input_chs),
            name=f"{name}-{0:02d}",
            **conv_specs,
        )
        self.layers_ = [layer]
        conv_specs.update(kernel_size=1)
        for l, n_filters in enumerate(filter_spec[1:]):
            layer = Conv2D(n_filters, name=f"{name}-{l+1:02d}", **conv_specs)
            self.layers_.append(layer)

    def call(self, inputs, training=False):
        """ Implements the feed forward part of the network """
        x = inputs
        for layer in self.layers_[:-1]:
            x = layer(x)
            x = sigmoid(x)
        x = self.layers_[-1](x)
        if self.decoder:
            x = tanh(x)
        else:
            x = sigmoid(x)
        return x
