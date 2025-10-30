import os

from sklearn.decomposition import PCA
import numpy as np
import keras
from sklearn.preprocessing import minmax_scale
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.layers import Dense, Dropout, Input, Concatenate, Activation, BatchNormalization
from keras.models import Sequential, Model, load_model
from keras import regularizers, optimizers, losses, layers

class Custom_Dropout(layers.Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=False):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        
        return inputs

class Custom_Loss(losses.Loss):
    def __init__(self, name="Custom_Loss"):
        super().__init__(name=name)
        # self.x_values = x_values


    def call(self, y_true, y_pred):
        # Check and handle shape mismatch
        # if y_true.shape != y_pred.shape:
        #     y_true = tf.reshape(y_true, y_pred.shape)

        # # Check and handle undefined tensors
        # y_true = tf.where(tf.math.is_finite(y_true), y_true, tf.zeros_like(y_true))
        ###
        n_dims = np.shape(y_true)[1]-1
        # b_size = 64#np.shape(y_true)[0]

        y_weight = y_true[:, n_dims:n_dims+1]#tf.gather(y_true, [n_dims, n_dims], axis=1)
        y_true = y_true[:, 0:n_dims]
        ###

        # Define weights
        # y_rand = self.x_values

        # Calculate loss
        loss = tf.norm(tf.square(y_true - y_pred), ord=1, axis=1)

        # res = tf.norm(loss, ord=2, axis=1)#(-y_true * tf.math.log(y_pred) - (tf.math.subtract(1.0, y_true)) * tf.math.log(tf.math.subtract(1.0, y_pred)))#mse(y_true, y_pred)#
        loss = tf.math.multiply(loss, y_weight)
        # loss = bce(y_true, y_pred)

        return tf.reduce_mean(loss)

class NNInv:
    def __init__(
        self,
        init=PCA(n_components=2),
        # size="medium",
        # style="bottleneck",
        loss="mean_squared_error",
        opt=keras.optimizers.Adam(learning_rate=0.001),
        l1=0.0,
        l2=0.01,
        dropout=False,
        latent_dims=2,
        verbose=1,
        **kwargs,
    ):
        self.stop = EarlyStopping(
            verbose=0, min_delta=0.001, mode="min", patience=50, restore_best_weights=True
        )
        self.callbacks = [self.stop]
        self.verbose = verbose
        self.init = init
        self.dropout = dropout
        self.opt = opt
        # self.epochs = epochs
        self.loss = loss
        self.l1 = l1
        self.l2 = l2
        self.latent_dims = latent_dims

        self.inv = None
        self.is_fitted = False
        K.clear_session()

    def fit(self, X, y=None, epochs=300, **kwargs):
        main_input = Input(shape=(self.latent_dims,), name="main_input")
        x = Dense(
            2048,
            activation="relu",
            # kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l1",
        )(main_input)
        
        x = Dense(
            2048,
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l2",
        )(x)
        
        x = Dense(
            2048,
            activation="relu",
            # kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l3",
        )(x)
        
        x = Dense(
            2048,
            activation="relu",
            # kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l4",
        )(x)
        
        x = Dense(
            y.shape[1],
            activation="sigmoid",
            kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="output",
        )(x)

        if self.dropout:
            x = Dropout(0.5)(x)


        self.model = Model(inputs=main_input, outputs=x)

        self.model.summary()

        self.model.compile(loss=self.loss, optimizer=self.opt)


        self.model.fit(
            X,
            y,
            batch_size=256,
            epochs=epochs,
            verbose=self.verbose,
            validation_split=0.05,
            callbacks=self.callbacks,
            **kwargs,
        )

        #this is unnecessary, but to fit into my pipeline it had to be done I guess
        encoded_input = Input(shape=(self.latent_dims,))
        l = self.model.get_layer("l1")(encoded_input)
        l = self.model.get_layer("l2")(l)
        l = self.model.get_layer("l3")(l)
        l = self.model.get_layer("l4")(l)
        decoder_layer = self.model.get_layer("output")(l)

        self.inv = Model(encoded_input, decoder_layer)

        self.is_fitted = True

    def fit_random(self, X, y, X_rand, epochs=300, **kwargs):    
        
        main_input = Input(shape=(self.latent_dims,), name="main_input")
        second_input = Input(shape=(2,), name="second_input")
        x = Concatenate(axis=1)([main_input, second_input])
        # x = Custom_Dropout(0.1, name='do1')(x)
        x = Dense(
            2048,
            activation="relu",
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l1",
        )(x)
        x = Dense(
            2048,
            activation="relu",
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l2",
        )(x)
        # x = Custom_Dropout(0.25, name='do3')(x)
        x = Dense(
            2048,
            activation="relu",
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l3",
        )(x)
        x = Dense(
            2048,
            activation="relu",
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l4",
        )(x)
        x = Dense(
            y.shape[1],
            activation="sigmoid",
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="output",
        )(x)

        # if self.dropout:
        # x = Dropout(0.5)(x)


        self.model = Model(inputs=[main_input, second_input], outputs=x)#

        custom_loss = Custom_Loss()

        self.model.compile(loss=custom_loss, optimizer=self.opt, metrics=['accuracy'])#custom_loss

        # self.fwdModel(inputs=[main_input, second_input], outputs=x)#

        rand_norm = tf.norm(X_rand-0.5, ord=1, axis=1)
        sample_weight = (rand_norm - np.min(rand_norm))/(np.max(rand_norm) - np.min(rand_norm))
        sample_weight = 0.25+np.abs(np.log(0.25+0.5*sample_weight))
        sample_weight = sample_weight.reshape(np.shape(rand_norm)[0],1)
        y = np.concatenate((y, np.ones((X.shape[0],1))), axis=1)#sample_weight

        self.model.fit(
            [X, X_rand],
            y,
            batch_size=128,
            epochs=epochs,
            verbose=self.verbose,
            validation_split=0.1,
            callbacks=self.callbacks,
            # sample_weight=sample_weight,
            shuffle=True,
            **kwargs
        )
        self.is_fitted = True

        encoded_input = Input(shape=(self.latent_dims+2,))#
        # l = self.model.get_layer("do1")(encoded_input)
        l = self.model.get_layer("l1")(encoded_input)
        # l = self.model.get_layer("bn1")(l)
        l = self.model.get_layer("l2")(l)
        # l = self.model.get_layer("bn2")(l)
        # l = self.model.get_layer("do3")(l)
        l = self.model.get_layer("l3")(l)
        # l = self.model.get_layer("bn3")(l)
        l = self.model.get_layer("l4")(l)
        # l = self.model.get_layer("bn4")(l)
        decoder_layer = self.model.get_layer("output")(l)
        self.inv = Model(inputs=encoded_input, outputs=decoder_layer)

        return

    def _is_fit(self):
        if self.is_fitted:
            return True
        else:
            raise Exception("Model not trained. Call fit() before calling transform()")

    def inverse_transform(self, X):
        
        if self._is_fit():
            return self.inv.predict(X)
        
    def save_weights(self, export_path: str):
        # Route `save_weights` to specific models.
        self.inv.save(os.path.join(export_path, "inv"))


    def load_weights(self, export_path: str):
        # Same for `load_weights`
        self.is_fitted = True
        self.inv = load_model(os.path.join(export_path, "inv"))
