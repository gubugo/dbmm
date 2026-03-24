# code.models.pytorch.nninv.py

###
### SSNP model pytorch implementation
###

# imports
import torch
import torch.nn as nn
import torch.optim as optim


class SSNP(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dims=2,
        n_classes=2,
        act="relu",
        bottleneck_activation="tanh",
        lr=1e-3,
        device=None,
        verbose=1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dims = latent_dims
        self.n_classes = n_classes
        self.verbose = verbose

        # device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # activations
        act_fn = nn.ReLU if act == "relu" else nn.Tanh
        bottleneck_fn = nn.Identity if bottleneck_activation == "linear" else nn.Tanh

        # ===== encoder =====
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            act_fn(),
            nn.Linear(512, 128),
            act_fn(),
            nn.Linear(128, 32),
            act_fn(),
            nn.Linear(32, latent_dims),
            bottleneck_fn(),
        )

        # ===== decoder shared trunk =====
        self.decoder_hidden = nn.Sequential(
            nn.Linear(latent_dims, 32),
            act_fn(),
            nn.Linear(32, 128),
            act_fn(),
            nn.Linear(128, 512),
            act_fn(),
        )

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
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

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
                nn.init.xavier_uniform_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0001)
    
    def forward(self, x):
        z = self.encoder(x)
        h = self.decoder_hidden(z)
        x_hat = self.decoder_out(h)
        logits = self.classifier(h)
        return z, x_hat, logits
    
    def forward_4d(self, x1, x2):
        z = self.encoder(x1)
        z_ = torch.cat((z,x2), dim=0)
        h = self.decoder_hidden(z_)
        x_hat = self.decoder_out(h)
        logits = self.classifier(h)
        return z, x_hat, logits

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
            total_loss = 0.0

            for xb, yb in loader:
                _, x_hat, logits = self.forward(xb)
                loss_cls = self.class_loss(logits, yb)
                loss_rec = self.recon_loss(x_hat, xb)
                loss = loss_cls + loss_rec

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if self.verbose:
                print(
                    f"epoch {epoch+1:03d} | loss = {total_loss/len(loader):.6f}"
                )

        self.is_fitted = True

    def fit_4d(
        self, 
        X1, 
        X2,
        y, 
        epochs=10, 
        batch_size=256
    ):
        X1 = torch.tensor(X1, dtype=torch.float32).to(self.device)
        X2 = torch.tensor(X2, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)

        dataset = torch.utils.data.TensorDataset(X1, X2, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            total_loss = 0.0

            for x1b, x2b, yb in loader:
                _, x_hat, logits = self.forward_4d(x1b, x2b)
                loss_cls = self.class_loss(logits, yb)
                loss_rec = self.recon_loss(x_hat, x1b)
                loss = loss_cls + loss_rec

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if self.verbose:
                print(
                    f"epoch {epoch+1:03d} | loss = {total_loss/len(loader):.6f}"
                )

        self.is_fitted = True

    # ==========================================================
    # api compatibility with sklearn / tensorflow version
    # ==========================================================
    def transform(self, X):
        self._check_fit()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            z = self.encoder(X)
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
            _, _, logits = self.forward(X)
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