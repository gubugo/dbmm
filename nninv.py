import os

from sklearn.decomposition import PCA

import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.layers import Dense, Dropout, Input, Concatenate, Activation
from keras.models import Sequential, Model, load_model
from keras import regularizers, optimizers


class NNInv:
    def __init__(
        self,
        init=PCA(n_components=2),
        # size="medium",
        # style="bottleneck",
        loss="mean_squared_error",
        opt=keras.optimizers.Adam(learning_rate=0.001),
        l1=0.0,
        l2=0.1,
        dropout=False,
        latent_dims=2,
        verbose=1,
        **kwargs,
    ):
        self.stop = EarlyStopping(
            verbose=0, min_delta=0.00001, mode="min", patience=20, restore_best_weights=True
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
        main_input = Input(shape=(self.latent_dims,), name="main_input", dtype=tf.float64)
        x = Dense(
            2048,
            # kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l1", 
            dtype=tf.float64,
        )(main_input)
        x = Activation("relu", dtype='float64', name='a1')(x)
        x = Dense(
            2048,
            # kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l2", 
            dtype=tf.float64,
        )(x)
        x = Activation("relu", dtype='float64', name='a2')(x)
        x = Dense(
            2048,
            # kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l3", 
            dtype=tf.float64,
        )(x)
        x = Activation("relu", dtype='float64', name='a3')(x)
        x = Dense(
            2048,
            # kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l4", 
            dtype=tf.float64,
        )(x)
        x = Activation("relu", dtype='float64', name='a4')(x)
        x = Dense(
            y.shape[1],
            # activation=Activation("sigmoid", dtype='float64'),
            kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="output", 
            dtype=tf.float64,
        )(x)
        x = Activation("sigmoid", dtype='float64', name='a5')(x)

        if self.dropout:
            x = Dropout(0.5)(x)


        self.model = Model(inputs=main_input, outputs=x)

        self.model.summary()

        self.model.compile(loss=self.loss, optimizer=self.opt)


        self.model.fit(
            X,
            y,
            batch_size=64,
            epochs=epochs,
            verbose=self.verbose,
            validation_split=0.05,
            callbacks=self.callbacks,
            **kwargs,
        )

        #this is unnecessary, but to fit into my pipeline it had to be done I guess
        encoded_input = Input(shape=(self.latent_dims,))
        l = self.model.get_layer("l1")(encoded_input)
        l = self.model.get_layer("a1")(encoded_input)
        l = self.model.get_layer("l2")(l)
        l = self.model.get_layer("a2")(l)
        l = self.model.get_layer("l3")(l)
        l = self.model.get_layer("a3")(l)
        l = self.model.get_layer("l4")(l)
        l = self.model.get_layer("a4")(l)
        l = self.model.get_layer("output")(l)
        decoder_layer = self.model.get_layer("a5")(l)

        self.inv = Model(encoded_input, decoder_layer)

        self.is_fitted = True

    def fit_random(self, X, y=None, epochs=300, **kwargs):
        X_rand = tf.random.Generator.from_seed(360).normal(stddev=1, shape=(X.shape[0],2), dtype=tf.float32)

        main_input = Input(shape=(self.latent_dims,), name="main_input", dtype=tf.float64)
        second_input = Input(shape=(2,), name="second_input", dtype=tf.float64)
        x = Concatenate(axis=1)([main_input, second_input])
        x = Dense(
            2048,
            activation="relu",
            # kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l1",
            dtype=tf.float64
        )(x)
        x = Dense(
            2048,
            activation="relu",
            # kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l2",
            dtype=tf.float64
        )(x)
        x = Dense(
            2048,
            activation="relu",
            # kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l3",
            dtype=tf.float64
        )(x)
        x = Dense(
            2048,
            activation="relu",
            # kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="l4",
            dtype=tf.float64
        )(x)
        x = Dense(
            y.shape[1],
            activation="sigmoid",
            kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
            name="output",
            dtype=tf.float64
        )(x)

        if self.dropout:
            x = Dropout(0.5)(x)


        self.model = Model(inputs=[main_input, second_input], outputs=x)

        self.model.compile(loss=self.loss, optimizer=self.opt)

        # self.fwd = Model(inputs=[main_input, second_input], outputs=x)
        self.model.fit(
            [X, X_rand],
            y,
            batch_size=16,
            epochs=epochs,
            verbose=self.verbose,
            validation_split=0.05,
            callbacks=self.callbacks,
            **kwargs
        )
        self.is_fitted = True

        #this is unnecessary, but to fit into my pipeline it had to be done I guess
        encoded_input = Input(shape=(self.latent_dims+2,))
        l = self.model.get_layer("l1")(encoded_input)
        l = self.model.get_layer("l2")(l)
        l = self.model.get_layer("l3")(l)
        l = self.model.get_layer("l4")(l)
        decoder_layer = self.model.get_layer("output")(l)
        self.inv = Model(encoded_input, decoder_layer)

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
