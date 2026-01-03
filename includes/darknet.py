"""Darknet-53 for yolo v3.
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, ZeroPadding2D
from tensorflow.keras.layers import add, Activation, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.regularizers import l2
import os

def conv2d_unit(x, filters, kernels, strides=1, number=0):
    """Convolution Unit
    This function defines a 2D convolution operation with BN and LeakyReLU.
    # Arguments
        x: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernels: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and
            height. Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
            Output tensor.
    """
    x = Conv2D(filters, kernels,
               padding='same',
               strides=strides,
               activation='linear',
               use_bias=True,
               kernel_regularizer=l2(5e-4),
               name="conv2d_%d"%number)(x)
    x = BatchNormalization(name="BN_%d"%number)(x)
    x = LeakyReLU(alpha=0.1,name="LRELU_%d"%number)(x)

    return x, number+1


def residual_block(inputs, filters, number=0):
    """Residual Block
    This function defines a 2D convolution operation with BN and LeakyReLU.
    # Arguments
        x: Tensor, input tensor of residual block.
        kernels: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
    # Returns
        Output tensor.
    """
    x, number = conv2d_unit(inputs, filters, (1, 1), number=number)
    x, number = conv2d_unit(x, 2 * filters, (3, 3), number=number)
    x = add([inputs, x], name="add_%d"%number)
    x = Activation('linear', name="act_%d"%number)(x)

    return x, number


def stack_residual_block(inputs, filters, n, number=0):
    """Stacked residual Block
    """
    x, number = residual_block(inputs, filters, number=number)

    for i in range(n - 1):
        x, number = residual_block(x, filters, number=number)

    return x, number


def darknet_base(inputs):
    """Darknet-53 base model.
    """

    x, number = conv2d_unit(inputs, 32, (3, 3), number=0)

    # x = ZeroPadding2D(padding=(1,1))(x)
    x, number = conv2d_unit(x, 64, (3, 3), strides=2, number=number)
    x, number = stack_residual_block(x, 32, n=1, number=number)

    # x = ZeroPadding2D(padding=(1,1))(x)
    x, number = conv2d_unit(x, 128, (3, 3), strides=2, number=number)
    x, number = stack_residual_block(x, 64, n=2, number=number)

    # x = ZeroPadding2D(padding=(1,1))(x)
    x, number = conv2d_unit(x, 256, (3, 3), strides=2, number=number)
    x, number = stack_residual_block(x, 128, n=8, number=number)

    # x = ZeroPadding2D(padding=(1,1))(x)
    x, number = conv2d_unit(x, 512, (3, 3), strides=2, number=number)
    x, number = stack_residual_block(x, 256, n=8, number=number)

    # x = ZeroPadding2D(padding=(1,1))(x)
    x, number = conv2d_unit(x, 1024, (3, 3), strides=2, number=number)
    x, number = stack_residual_block(x, 512, n=4, number=number)

    return x


def darknet(inputs, load=False):
    """Darknet-53 classifier.
    """
    x = darknet_base(inputs)

    #x = GlobalAveragePooling2D()(x)
    #x = Dense(1000, activation='softmax')(x)

    model = Model(inputs, x, name="Darknet53")
    print(__file__)
    if load:
        model.load_weights(os.path.join(os.path.split(__file__)[0], "darknet53_weights.h5"))

    return model


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-10.1/lib64;/usr/local/cuda-10.1/extras/CUPTI/lib64"
    os.environ["PATH"] += ":/usr/local/cuda-10.1/bin"
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    model = darknet(Input(shape=(512, 512, 3)))
    print(model.summary())