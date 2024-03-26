from tensorflow.keras.layers import Layer, Activation, Conv2D
import tensorflow.keras.backend as K1
import tensorflow as tf



class PAM(Layer):
    def __init__(self,
                 # gamma_initializer=tf.zeros_initializer(),
                 # gamma_regularizer=None,
                 # gamma_constraint=None,
                 **kwargs):

        super(PAM, self).__init__(**kwargs)

        self.b = Conv2D(6, 1, use_bias=False, kernel_initializer='he_normal')
        self.c = Conv2D(6, 1, use_bias=False, kernel_initializer='he_normal')
        self.d = Conv2D(50, 1, use_bias=False, kernel_initializer='he_normal')

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer='zeros',
                                     regularizer=None,
                                     constraint=None,
                                     name='gamma',
                                     trainable=True
                                     )

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        # print('chengwei test inputshape:', input.shape)
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape

        b = self.b(input)
        c = self.c(input)
        d = self.d(input)

        vec_b = K1.reshape(b, (-1, h * w, filters // 8))
        vec_cT = tf.transpose(K1.reshape(c, (-1, h * w, filters // 8)), (0, 2, 1))
        bcT = K1.batch_dot(vec_b, vec_cT)
        softmax_bcT = Activation('softmax')(bcT)
        vec_d = K1.reshape(d, (-1, h * w, filters))
        bcTd = K1.batch_dot(softmax_bcT, vec_d)
        bcTd = K1.reshape(bcTd, (-1, h, w, filters))

        out = self.gamma*bcTd + input
        return out

class CAM(Layer):
    def __init__(self,
                 # gamma_initializer=tf.zeros_initializer(),
                 # gamma_regularizer=None,
                 # gamma_constraint=None,
                 **kwargs):

        super(CAM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer='zeros',
                                     regularizer=None,
                                     constraint=None,
                                     name='gamma',
                                     trainable=True
                                     )

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        input_shape = input.get_shape().as_list()
        _, filters, hw = input_shape

        vec_b = tf.transpose(input, (0, 2, 1))
        vec_cT = input
        bcT = K1.batch_dot(vec_b, vec_cT)
        softmax_bcT = Activation('softmax')(bcT)
        vec_d = tf.transpose(input, (0, 2, 1))
        bcTd = K1.batch_dot(softmax_bcT, vec_d)
        bcTd = tf.transpose(bcTd, (0, 2, 1))

        out = self.gamma*bcTd + input
        return out

