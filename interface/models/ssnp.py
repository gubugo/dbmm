import gc
import os

import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, minmax_scale
from keras import backend as K
from keras import datasets as kdatasets
from keras import losses, optimizers, regularizers, layers
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.layers import Dense, Dropout, Input, Concatenate, BatchNormalization
from keras.models import Model, Sequential, load_model
# os.environ["TF_DETERMINISTIC_OPS"] = "1"

from keras.losses import binary_crossentropy as k_bce

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
        #mse
        loss = tf.square(y_true - y_pred)

        #bce
        # loss = -y_true*tf.math.log(y_pred)-(1-y_true)*tf.math.log(1-y_pred)

        # res = tf.norm(loss, ord=2, axis=1)#(-y_true * tf.math.log(y_pred) - (tf.math.subtract(1.0, y_true)) * tf.math.log(tf.math.subtract(1.0, y_pred)))#mse(y_true, y_pred)#
        loss = tf.math.multiply(loss, y_weight)
        # loss = bce(y_true, y_pred)

        return tf.reduce_mean(loss)


class SSNP:
    def __init__(
        self,
        init_labels="precomputed",
        # epochs=100,
        input_l1=0.0,
        input_l2=0.0,
        bottleneck_l1=0.0,
        bottleneck_l2=0.0,
        latent_dims=2,
        verbose=1,
        opt="adam",
        bottleneck_activation="tanh",
        act="relu",
        init="glorot_uniform",
        bias=0.0001,
        patience=3,
        min_delta=0.01,
    ):
        self.init_labels = init_labels
        self.verbose = verbose
        self.opt = opt
        self.act = act
        self.init = init
        self.bias = bias
        self.input_l1 = input_l1
        self.input_l2 = input_l2
        self.bottleneck_l1 = bottleneck_l1
        self.bottleneck_l2 = bottleneck_l2
        self.latent_dims = latent_dims
        self.bottleneck_activation = bottleneck_activation
        self.patience = patience
        self.min_delta = min_delta

        self.label_bin = LabelBinarizer()

        self.fwd = None
        self.inv = None

        tf.random.set_seed(42)

        self.is_fitted = False
        K.clear_session()

    

    def fit(self, X, y=None, X_rand=tf.zeros((1,1)), epochs=0):
        # pca = PCA(n_components=2)
        # X_rand = pca.fit_transform(X)
        # X_rand = minmax_scale(X_rand, feature_range=(-1,1))

        if y is None and self.init_labels == "precomputed":
            raise Exception("Must provide labels when using init_labels = precomputed")

        if y is None:
            y = self.init_labels.fit_predict(X)

        expre = tf.equal(X_rand, tf.zeros((1,1))).numpy().all()
        X_rand  = tf.zeros((X.shape[0],0)) if expre else X_rand
        ext_dim = 0 if expre else 2

        self.label_bin.fit(y)

        main_input = Input(shape=(X.shape[1],), name="main_input")
        second_input = Input(shape=(ext_dim,), name="second_input")
        # print(main_input)
        x = Dense(
            512,
            activation=self.act,
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )(main_input)
        x = Dense(
            128,
            activation=self.act,
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )(x)
        x = Dense(
            32,
            activation=self.act,
            # activity_regularizer=regularizers.l1_l2(l1=self.input_l1, l2=self.input_l2),
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )(x)
        encoded = Dense(
            self.latent_dims,
            activation=self.bottleneck_activation,
            # kernel_regularizer=regularizers.l1_l2(l1=self.bottleneck_l1, l2=self.bottleneck_l2),
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )(x)

        concat = Concatenate(axis=1)([encoded, second_input])
        x = Dense(
            32,
            activation=self.act,
            kernel_initializer=self.init,
            name="enc1",
            bias_initializer=Constant(self.bias),
        )(concat)
        # x = BatchNormalization(name="bn1")(x)
        # x = Custom_Dropout(0.1, name='do1')(x)
        x = Dense(
            128,
            activation=self.act,
            kernel_initializer=self.init,
            name="enc2",
            bias_initializer=Constant(self.bias),
        )(x)
        # x = BatchNormalization(name="bn2")(x)
        x = Dense(
            512,
            activation=self.act,
            kernel_initializer=self.init,
            name="enc3",
            bias_initializer=Constant(self.bias),
        )(x)
        # x = BatchNormalization(name="bn3")(x)
        n_classes = len(np.unique(y))

        if n_classes == 2:
            n_units = 1
        else:
            n_units = n_classes

        main_output = Dense(
            n_units,
            activation="softmax",
            name="main_output",
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )(x)

        decoder_output = Dense(
            X.shape[1],
            activation="sigmoid",
            name="decoder_output",
            kernel_initializer=self.init,
            bias_initializer=Constant(self.bias),
        )(x)

        model = Model(inputs=[main_input, second_input], outputs=[main_output, decoder_output])

        custom_loss = Custom_Loss()

        model.compile(
            optimizer=self.opt,
            loss={
                "main_output": "categorical_crossentropy",
                "decoder_output": custom_loss,
            },
            metrics=["accuracy"],
        )

        if self.patience > 0:
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    min_delta=self.min_delta,
                    patience=self.patience,
                    restore_best_weights=True,
                    verbose=self.verbose,
                )
            ]
        else:
            callbacks = []

        
        rand_norm = tf.norm(X_rand, ord=1, axis=1)
        sample_weight = (rand_norm - np.min(rand_norm))/(np.max(rand_norm) - np.min(rand_norm))
        sample_weight = 0.25+np.abs(np.log(0.25+0.5*sample_weight))
        sample_weight = sample_weight.reshape(np.shape(rand_norm)[0],1)
        X_res = np.concatenate((X, np.ones((X.shape[0],1))), axis=1)#sample_weight
        
        hist = model.fit(
            [X, X_rand],
            [self.label_bin.transform(y), X_res],
            batch_size=256,  # TODO: (UNCHANGE)
            epochs=epochs,
            shuffle=True,
            verbose=self.verbose,
            validation_split=0.05,
            callbacks=callbacks,
        )
        
        encoded_input = Input(shape=(self.latent_dims+ext_dim,))
        l = model.get_layer("enc1")(encoded_input)
        # l = model.get_layer("do1")(l)
        l = model.get_layer("enc2")(l)
        # l = model.get_layer("bn2")(l)
        l = model.get_layer("enc3")(l)
        # l = model.get_layer("bn3")(l)
        decoder_layer = model.get_layer("decoder_output")(l)
        classifier_layer = model.get_layer("main_output")(l)

        self.inv = Model(encoded_input, decoder_layer)

        self.fwd = Model(inputs=main_input, outputs=encoded)
        self.clustering = Model(inputs=[main_input, second_input], outputs=main_output)
        self.latent_clustering = Model(inputs=encoded_input, outputs=classifier_layer)
        self.is_fitted = True

        return hist

    def transform(self, X):
        if self._is_fit():
            return self.fwd.predict(X)

    def inverse_transform(self, X_2d):
        if self._is_fit():
            return self.inv.predict(X_2d)

    def predict(self, X):
        if self._is_fit():
            y_pred = self.clustering.predict(X)
            return self.label_bin.inverse_transform(y_pred)

    def _is_fit(self):
        if self.is_fitted:
            return True
        else:
            raise Exception("Model not trained. Call fit() before calling transform()")
        
    def save_weights(self, export_path: str):
        # Route `save_weights` to specific models.
        self.fwd.save(os.path.join(export_path, "fwd"))
        self.inv.save(os.path.join(export_path, "inv"))
        # self.clustering.save(os.path.join(export_path, "clustering"))
        # self.latent_clustering.save(os.path.join(export_path, "latent_clustering"))


    def load_weights(self, export_path: str):
        # Same for `load_weights`
        self.is_fitted = True
        self.fwd = load_model(os.path.join(export_path, "fwd"))
        self.inv = load_model(os.path.join(export_path, "inv"))
        # self.clustering = load_model(os.path.join(export_path, "clustering"))
        # self.latent_clustering = load_model(os.path.join(export_path, "latent_clustering"))

