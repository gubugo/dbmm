from sklearn import decomposition, preprocessing
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers, optimizers


class NNInv:
    def __init__(
        self,
        init=decomposition.PCA(n_components=2),
        # size="medium",
        # style="bottleneck",
        loss="mean_squared_error",
        opt="adam",
        l1=0.0,
        l2=0.0,
        dropout=False,
        latent_dims=2
    ):
        self.stop = EarlyStopping(
            verbose=0, min_delta=0.00001, mode="min", patience=10, restore_best_weights=True
        )
        self.callbacks = [self.stop]

        self.init = init
        self.dropout = dropout
        self.opt = opt
        # self.epochs = epochs
        self.loss = loss
        self.l1 = l1
        self.l2 = l2
        self.latent_dims = latent_dims

        self.is_fitted = False
        K.clear_session()

    def fit(self, X, y=None, epochs=300):
        main_input = Input(shape=(self.latent_dims,), name="main_input")
        x = Dense(
            2048,
            activation="relu",
            kernel_initialize="he_uniform",
            bias_initializer=Constant(0.01),
        )(main_input)
        x = Dense(
            2048,
            activation="relu",
            kernel_initialize="he_uniform",
            bias_initializer=Constant(0.01),
        )(main_input)
        x = Dense(
            2048,
            activation="relu",
            kernel_initialize="he_uniform",
            bias_initializer=Constant(0.01),
        )(main_input)
        x = Dense(
            2048,
            activation="relu",
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
        )(x)
        x = Dense(
            y.shape[1],
            activation="sigmoid",
            kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
            kernel_initializer="he_uniform",
            bias_initializer=Constant(0.01),
        )(x)

        if self.dropout:
            x = Dropout(0.5)(x)


        self.model = Model(inputs=main_input, outputs=x)

        self.model.compile(loss=self.loss, optimizer=self.opt)


        self.model.fit(
            X,
            y,
            batch_size=32,
            epochs=epochs,
            verbose=0,
            validation_split=0.05,
            callbacks=self.callbacks,
        )
        self.is_fitted = True

    def _is_fit(self):
        if self.is_fitted:
            return True
        else:
            raise Exception("Model not trained. Call fit() before calling transform()")

    def inverse_transform(self, X):
        if self._is_fit():
            return self.model.predict(X)
