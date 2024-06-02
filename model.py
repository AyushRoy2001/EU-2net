import tensorflow as tf
import tensorflow.keras as keras
import keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

def wavelet(feature_map, order):
    B,H,W,C = feature_map.shape
    wav = tf.image.sobel_edges(feature_map)
    wav1 = wav[:,:,:,0]
    wav2 = wav[:,:,:,1]
    wav = tf.keras.layers.Concatenate()([wav1,wav2])
    wav = tf.keras.layers.Conv2D(2*C, (3,3), (1,1),padding="same")(wav) 
    wav = tf.keras.layers.Conv2D(C, (1,1), (1,1),padding="same", name=order)(wav) 
    return wav 

def SE(x):
    x = Conv2D(32, 3, padding="same")(x)
    x1 = tf.keras.layers.GlobalAveragePooling2D()(x)
    x1 = tf.keras.layers.Dense(8, activation='relu')(x1)
    x1 = tf.keras.layers.Dense(32, activation='sigmoid')(x1)
    x1 = tf.keras.layers.Reshape((1, 1, 32))(x1)
    x = tf.keras.layers.Multiply()([x, x1])
    return x

def FSiAM(x):
    x1 = tf.keras.layers.Permute((2, 1))(x)
    x2 = tf.matmul(x1, x)
    x2 = tf.keras.layers.GlobalAveragePooling1D()(x2)
    x2 = tf.keras.activations.sigmoid(x2)
    x2 = tf.expand_dims(x2, axis=1)
    x2 = tf.expand_dims(x2, axis=1)
    return x2

class EnsembleLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EnsembleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(EnsembleLayer, self).build(input_shape)
        # trainable weight
        self.weights_variable = tf.Variable(
            initial_value=tf.ones((6,), dtype=tf.float32),
            trainable=True,
            name='weights'
        )

    def call(self, inputs):
        y1, y2, y3, y4, y5, y6 = inputs
        stacked_masks = tf.stack([y1, y2, y3, y4, y5, y6], axis=-1)
        normalized_weights = tf.nn.softmax(self.weights_variable) # normalize weights between 0 and 1
        final_mask = tf.reduce_sum(stacked_masks * tf.expand_dims(normalized_weights, axis=0), axis=-1)
        #final_mask = tf.where(final_mask > 0.5, 1.0, 0.0)
        return final_mask

    def compute_output_shape(self, input_shape):
        # The output shape is the same as the input shape of the masks (height, width, num_classes)
        return input_shape[0][:-1] + (1,)

    def get_config(self):
        config = super().get_config()
        return config
    
def conv_block(inputs, out_ch, rate=1):
    x = SeparableConv2D(out_ch, 3, padding="same", dilation_rate=1)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def RSU_L(inputs, out_ch, int_ch, num_layers, rate=2):
    """ Initial Conv """
    x = conv_block(inputs, out_ch)
    init_feats = x

    """ Encoder """
    skip = []
    x = conv_block(x, int_ch)
    skip.append(x)

    for i in range(num_layers-2):
        x = MaxPool2D((2, 2))(x)
        x = conv_block(x, int_ch)
        skip.append(x)

    """ Bridge """
    x = conv_block(x, int_ch, rate=rate)

    """ Decoder """
    skip.reverse()

    x = Concatenate()([x, skip[0]])
    x = conv_block(x, int_ch)

    for i in range(num_layers-3):
        x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x = Concatenate()([x, skip[i+1]])
        x = conv_block(x, int_ch)

    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = Concatenate()([x, skip[-1]])
    x = conv_block(x, out_ch)

    """ Add """
    x = Add()([x, init_feats])
    return x

def RSU_4F(inputs, out_ch, int_ch):
    """ Initial Conv """
    x0 = conv_block(inputs, out_ch, rate=1)

    """ Encoder """
    x1 = conv_block(x0, int_ch, rate=1)
    x2 = conv_block(x1, int_ch, rate=2)
    x3 = conv_block(x2, int_ch, rate=4)

    """ Bridge """
    x4 = conv_block(x3, int_ch, rate=8)

    """ Decoder """
    x = Concatenate()([x4, x3])
    x = conv_block(x, int_ch, rate=4)

    x = Concatenate()([x, x2])
    x = conv_block(x, int_ch, rate=2)

    x = Concatenate()([x, x1])
    x = conv_block(x, out_ch, rate=1)

    """ Addition """
    x = Add()([x, x0])
    return x

def u2net(input_shape, out_ch, int_ch, num_classes=1):
    """ Input Layer """
    inputs = Input(input_shape)
    s0 = inputs

    """ Encoder """
    s1 = RSU_L(s0, out_ch[0], int_ch[0], 7)
    p1 = MaxPool2D((2, 2))(s1)

    s2 = RSU_L(p1, out_ch[1], int_ch[1], 6)
    p2 = MaxPool2D((2, 2))(s2)

    s3 = RSU_L(p2, out_ch[2], int_ch[2], 5)
    p3 = MaxPool2D((2, 2))(s3)

    s4 = RSU_L(p3, out_ch[3], int_ch[3], 4)
    p4 = MaxPool2D((2, 2))(s4)

    s5 = RSU_4F(p4, out_ch[4], int_ch[4])
    p5 = MaxPool2D((2, 2))(s5)

    """ Bridge """
    b1 = RSU_4F(p5, out_ch[5], int_ch[5])
    b2 = UpSampling2D(size=(2, 2), interpolation="bilinear")(b1)

    """ Decoder """
    x = FSiAM(tf.keras.layers.Reshape((s5.shape[1]*s5.shape[2],s5.shape[3]))(s5))
    x = tf.keras.layers.Multiply()([s5, x])
    fft = wavelet(s5, "edge_1") 
    d1 = Concatenate(name="ONE")([SE(b2), x, fft])
    d1 = tf.keras.layers.SeparableConv2D(512,(3,3),(1, 1),padding="same")(d1)
    d1 = tf.keras.layers.SeparableConv2D(256,(1,1),(1, 1),padding="same")(d1)
    d1 = RSU_4F(d1, out_ch[6], int_ch[6])
    u1 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d1)
    
    x = FSiAM(tf.keras.layers.Reshape((s4.shape[1]*s4.shape[2],s4.shape[3]))(s4))
    x = tf.keras.layers.Multiply()([s4, x])
    fft = wavelet(s4, "edge_2")
    d2 = Concatenate(name="TWO")([SE(u1), x, fft])
    d2 = tf.keras.layers.SeparableConv2D(256,(3,3),(1, 1),padding="same")(d2)
    d2 = tf.keras.layers.SeparableConv2D(128,(1,1),(1, 1),padding="same")(d2)
    d2 = RSU_L(d2, out_ch[7], int_ch[7], 4)
    u2 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d2)
    
    x = FSiAM(tf.keras.layers.Reshape((s3.shape[1]*s3.shape[2],s3.shape[3]))(s3))
    x = tf.keras.layers.Multiply()([s3, x])
    fft = wavelet(s3, "edge_3")
    d3 = Concatenate(name="THREE")([SE(u2), x, fft])
    d3 = tf.keras.layers.SeparableConv2D(128,(3,3),(1, 1),padding="same")(d3)
    d3 = tf.keras.layers.SeparableConv2D(64,(1,1),(1, 1),padding="same")(d3)
    d3 = RSU_L(d3, out_ch[8], int_ch[8], 5)
    u3 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d3)

    x = FSiAM(tf.keras.layers.Reshape((s2.shape[1]*s2.shape[2],s2.shape[3]))(s2))
    x = tf.keras.layers.Multiply()([s2, x])
    fft = wavelet(s2, "edge_4")
    d4 = Concatenate(name="FOUR")([SE(u3), x, fft])
    d4 = tf.keras.layers.SeparableConv2D(64,(3,3),(1, 1),padding="same")(d4)
    d4 = tf.keras.layers.SeparableConv2D(32,(1,1),(1, 1),padding="same")(d4)
    d4 = RSU_L(d4, out_ch[9], int_ch[9], 6)
    u4 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d4)
    
    x = FSiAM(tf.keras.layers.Reshape((s1.shape[1]*s1.shape[2],s1.shape[3]))(s1))
    x = tf.keras.layers.Multiply()([s1, x])
    fft = wavelet(s1, "edge_5")
    d5 = Concatenate(name="FIVE")([SE(u4), x, fft])
    d5 = tf.keras.layers.SeparableConv2D(32,(3,3),(1, 1),padding="same")(d5)
    d5 = tf.keras.layers.SeparableConv2D(16,(1,1),(1, 1),padding="same")(d5)
    d5 = RSU_L(d5, out_ch[10], int_ch[10], 7)

    """ Side Outputs """
    y1 = Conv2D(num_classes, 3, padding="same")(d5)

    y2 = Conv2D(num_classes, 3, padding="same")(d4)
    y2 = UpSampling2D(size=(2, 2), interpolation="bilinear")(y2)

    y3 = Conv2D(num_classes, 3, padding="same")(d3)
    y3 = UpSampling2D(size=(4, 4), interpolation="bilinear")(y3)

    y4 = Conv2D(num_classes, 3, padding="same")(d2)
    y4 = UpSampling2D(size=(8, 8), interpolation="bilinear")(y4)

    y5 = Conv2D(num_classes, 3, padding="same")(d1)
    y5 = UpSampling2D(size=(16, 16), interpolation="bilinear")(y5)

    y6 = Conv2D(num_classes, 3, padding="same")(b1)
    y6 = UpSampling2D(size=(32, 32), interpolation="bilinear")(y6)

    y1 = Activation("sigmoid")(y1)
    y2 = Activation("sigmoid")(y2)
    y3 = Activation("sigmoid")(y3)
    y4 = Activation("sigmoid")(y4)
    y5 = Activation("sigmoid")(y5)
    y6 = Activation("sigmoid")(y6)
    
    y0 = EnsembleLayer(name="ensemble_layer")([y1, y2, y3, y4, y5, y6]) 

    model = tf.keras.models.Model(inputs, outputs=y0)
    return model

def build_u2net(input_shape, num_classes=1):
    out_ch = [64, 128, 256, 512, 512, 512, 512, 256, 128, 64, 64]
    int_ch = [32, 32, 64, 128, 256, 256, 256, 128, 64, 32, 16]
    model = u2net(input_shape, out_ch, int_ch, num_classes=num_classes)
    return model

model = build_u2net((512, 512, 1))
optimizer = Adam(lr=0.0001)
model.compile(loss=combined_loss, metrics=["accuracy",dice_score,recall,precision,iou], optimizer = optimizer)
model.summary()
    
