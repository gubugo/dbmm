# code.models.pytorch.sharp.py

###
### ShaRP model pytorch implementation
###

# imports
import torch
import torch.nn as nn
import torch.optim as optim

from code.models.pytorch.sampling_layers import get_layer_builder

class Encoder(nn.Module):
    def __init__(
        self,
        input_dims,
        bias=1e-4,
        act="relu",
    ):
        super().__init__()
        self.bias = bias

        # activations
        act_fn = nn.ReLU if act == "relu" else nn.Tanh

        self.encoder = nn.Sequential(
            nn.Linear(input_dims, 512),
            act_fn(),
            nn.Linear(512, 128),
            act_fn(),
            nn.Linear(128, 32),
            act_fn(),
        )

        # initializers
        self._init_weights()

    # ==========================================================
    # internal
    # ==========================================================
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain('relu')
                nn.init.xavier_uniform_(m.weight, gain=gain)
                nn.init.constant_(m.bias, self.bias)
    
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(
        self,
        input_dims,
        bias=1e-4,
        act="relu",
    ):
        super().__init__()
        self.bias = bias

        # activations
        act_fn = nn.ReLU if act == "relu" else nn.Tanh

        self.encoder = nn.Sequential(
            nn.Linear(input_dims, 32),
            nn.BatchNorm1d(32),
            act_fn(),
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            act_fn(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            act_fn(),
        )

        # initializers
        self._init_weights()

    # ==========================================================
    # internal
    # ==========================================================
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain('relu')
                nn.init.xavier_uniform_(m.weight, gain=gain)
                nn.init.constant_(m.bias, self.bias)
    
    def forward(self, x):
        return self.encoder(x)

class ShaRP(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim=2,
        n_classes=2,
        variational_layer=None,
        act="relu",
        bottleneck_activation="tanh",
        l1=0.0,
        l2=0.5,
        lr=1e-3,
        device=None,
        verbose=1,
    ):
        super().__init__()

        self.variational_layer = variational_layer
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.verbose = verbose

        # device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # activations
        act_fn = nn.ReLU if act == "relu" else nn.Tanh
        # bottleneck_fn = nn.Identity if bottleneck_activation == "linear" else nn.Tanh

        # ===== encoder =====
        self.encoder = Encoder(input_dims=input_dim)

        # ===== sampling layer =====
        self.variational = self._build_variational_layer(bottleneck_activation)

        # ===== decoder shared trunk ===== 
        self.decoder_hidden = Decoder(input_dims=2)

        # reconstruction head
        self.decoder_out = nn.Sequential(
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, n_classes),
            # nn.Softmax(),
        )

        # initializers
        self._init_weights()

        # optimizer 
        self.optimizer = optim.Adam([
            {'params': self.decoder_hidden.parameters(), 'weight_decay': 0.0},  # No L2 for these
            {'params': self.decoder_out.parameters(), 'weight_decay': 0.0},  # No L2 for these
            {'params': self.classifier.parameters(), 'weight_decay': 0.0},  # No L2 for these
            {'params': self.encoder.parameters(), 'weight_decay': 1e-4}, # L2 applied here
            {'params': self.variational.parameters(), 'weight_decay': 0.1} # L2 applied here
        ], lr=lr)

        # losses
        self.class_loss = nn.CrossEntropyLoss()
        self.recon_loss = nn.BCELoss()#nn.BCEWithLogitsLoss()

        self.to(self.device)
        self.is_fitted = False

    # ==========================================================
    # internal
    # ==========================================================
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain('relu')
                nn.init.xavier_uniform_(m.weight, gain=gain)
                nn.init.constant_(m.bias, 0.0001)

    def _build_variational_layer(self, act):
        if isinstance(self.variational_layer, str):
            return get_layer_builder(self.variational_layer)(
                self.latent_dim, act=act
            )
        else:
            return self.variational_layer(self.latent_dim)
    
    def forward(self, x):
        encoded = self.encoder(x)
        z_mean, z_log_var, z, kl_loss = self.variational(encoded)
        h = self.decoder_hidden(z)
        x_hat = self.decoder_out(h)
        logits = self.classifier(h)
        return z, x_hat, logits, kl_loss

    # ==========================================================
    # training
    # ==========================================================
    def fit(
        self, 
        X, 
        y, 
        epochs=10, 
        batch_size=256
    ):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0

            for xb, yb in loader:
                self.optimizer.zero_grad()

                _, x_hat, logits, loss_kl = self.forward(xb)

                loss_cls = self.class_loss(logits, yb)
                loss_rec = self.recon_loss(x_hat, xb)
                # print(loss_cls)
                # print(loss_rec)
                # print(loss_kl)
                loss = loss_cls + 3.0*loss_rec + loss_kl

                # ## for VERBOSE metrics
                # # Acc
                # pred = loss_cls.argmax(dim=1)
                # acc = (pred == yb).float().mean()
                # # mean absolute error
                # mae = torch.mean(torch.abs(loss_rec - xb))


                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()

            if self.verbose:
                print(
                    f"epoch {epoch+1:03d} | loss = {total_loss/len(loader):.6f}"
                )
        self.eval()
        self.is_fitted = True

    # ==========================================================
    # api compatibility with sklearn / tensorflow version
    # ==========================================================
    def transform(self, X):
        self._check_fit()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            encoded = self.encoder(X)
            _, _, z, _ = self.variational(encoded)
        return z.cpu().numpy()

    def inverse_transform(self, Z):
        self._check_fit()
        Z = torch.tensor(Z, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            h = self.decoder_hidden(Z)
            X_hat = self.decoder_out(h)
        return X_hat.cpu().numpy()

    def predict(self, X):
        self._check_fit()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, _, logits, _ = self.forward(X)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()

    # ==========================================================
    # utils
    # ==========================================================
    def _check_fit(self):
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call fit() first.")

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.is_fitted = True