# code.models.pytorch.nninv.py

###
### Neural Network Inversion (NNInv) model pytorch implementation
###

import torch
import torch.nn as nn
import torch.optim as optim
import copy


class NNInvTorch(nn.Module):
    def __init__(
        self,
        latent_dims=2,
        output_dim=None,
        lr=1e-3,
        l1=0.0,
        l2=0.0,
        dropout=False,
        device=None
    ):
        super().__init__()

        self.latent_dims = latent_dims
        self.output_dim = output_dim
        self.l1 = l1
        self.l2 = l2
        self.dropout = dropout

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        layers = [
            nn.Linear(latent_dims, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim),
            nn.Sigmoid()
        ]

        if dropout:
            layers.insert(-1, nn.Dropout(0.5))

        self.model = nn.Sequential(*layers)

        self._init_weights()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.to(self.device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        return self.model(x)

    def fit(
        self,
        X,
        y,
        epochs=300,
        batch_size=256,
        patience=20,
        val_split=0.2,
        verbose=True,
    ):
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        y = torch.as_tensor(y, dtype=torch.float32).to(self.device)

        n_val = int(len(X) * val_split)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

        best_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(epochs):
            self.train()

            perm = torch.randperm(len(X_train))
            for i in range(0, len(X_train), batch_size):
                idx = perm[i : i + batch_size]
                xb = X_train[idx]
                yb = y_train[idx]

                self.zero_grad()
                preds = self(xb)

                loss = self.criterion(preds, yb)
                if self.l1 > 0:
                    loss += self.l1 * sum(p.abs().sum() for p in self.parameters())
                if self.l2 > 0:
                    loss += self.l2 * sum(p.pow(2).sum() for p in self.parameters())

                loss.backward()
                self.optimizer.step()

            self.eval()
            with torch.no_grad():
                val_preds = self(X_val)
                val_loss = self.criterion(val_preds, y_val)

            if verbose:
                print(f"epoch {epoch+1:03d} | val_loss = {val_loss:.6f}")

            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = copy.deepcopy(self.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print("early stopping")
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

    def inverse_transform(self, X):
        self.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self(X).cpu().numpy()

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))