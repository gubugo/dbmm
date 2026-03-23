# code.models.pytorch.sampling_layers.py

###
### ShaRP model's sampling layers pytorch implementation
###

# imports
import torch as T
import torch.nn as nn
import torch.distributions as dist

def get_layer_builder(layer_name: str):
    return SamplingLayer.builder_for(layer_name)


class SamplingLayer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        act="tanh",
        init="glorot_uniform",
        bias=1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.act = act
        self.init = init
        self.bias = bias

    @staticmethod
    def builder_for(layer_name: str):
        if layer_name == "diagonal_normal":
            return DiagonalNormalSampling
        # elif layer_name == "centered_diagonal_normal":
        #     return CenteredDiagonalNormalSampling
        # elif layer_name == "student_t":
        #     return StudentTSampling
        # elif layer_name == "generalized_normal":
        #     return GeneralizedNormalSampling
        # elif layer_name == "triangle":
        #     return TriangleSampling
        # elif layer_name == "laplace":
        #     return LaplaceSampling
        # elif layer_name == "gumbel":
        #     return GumbelSampling
        # elif layer_name == "polygon":
        #     return PolygonSampling
        # elif layer_name == "spherical":
        #     return SphericalSampling
        else:
            raise ValueError(
                f"layer name {layer_name} does not correspond to an implementation"
            )

    def sample(self, inputs, training=None):
        raise NotImplementedError("override sample() in derived classes")

    def add_kl_loss(self, samples):
        raise NotImplementedError()

    def forward(self, inputs, training=None):
        z, kl_loss = self.sample(inputs, training=self.training)
        self.add_kl_loss(z)
        # print(T.mean(z, dim=0))
        # print(T.mean(z, dim=-1))
        return T.mean(z), T.log(T.var(z)), z, kl_loss

    def _layer_kwargs(self):
        return {
            "activation": self.act,
            "kernel_initializer": self.init,
            "bias_initializer": self.bias,
        }
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain('relu')
                nn.init.xavier_uniform_(m.weight, gain=gain)
                nn.init.constant_(m.bias, self.bias)

    # the **kwargs argument allows calls to this method to replace one of the default values.
    def make_dense_param_layer(self, n_params: int) -> nn.Sequential:
        act_fn = nn.Tanh if self.act == "tanh" else nn.Identity

        self.layer = nn.Sequential(
            nn.Linear(32, n_params),
            act_fn()
        )

        self._init_weights()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "latent_dim": self.latent_dim,
                "act": self.act,
                "init": self.init,
                "bias": self.bias,
            }
        )
        return config
    
class DiagonalNormalSampling(SamplingLayer):
    def __init__(
        self,
        latent_dim: int,
        prior_loc: float = 0.0,
        prior_scale: float = 1.0,
        kl_weight: float = 0.05,
        kl_mu_weight: float = 0,
        use_exact_kl: bool = True,
        act="tanh",
        name="diag_normal_sampling",
        **kwargs,
    ):
        super().__init__(
            latent_dim=latent_dim,
            act=act,
            **kwargs,
        )
        self.prior_loc = prior_loc
        self.prior_scale = prior_scale
        self.kl_weight = kl_weight
        self.kl_mu_weight = kl_mu_weight
        self.use_exact_kl = use_exact_kl
        # self.sampler = dist.Independent(dist.Normal(mu, sigma), 1)

        # self.prior = dist.Independent(
        #     dist.Normal(loc=self.prior_loc, scale=self.prior_scale * T.ones(self.latent_dim)),
        #     1
        # )
        # self.prior = MultivariateNormal(
        #     loc=self.prior_loc,
        #     covariance_matrix=T.diag_embed(self.prior_scale**2)
        # )

        self.dense_params = self.make_dense_param_layer(
            self.latent_dim*2,
        )

    def sample(self, x, training):
        params = self.layer(x)  # shape: [batch, 2*latent_dim]

        # Split into mean and scale
        mu = params[..., :self.latent_dim]
        scale_raw = params[..., self.latent_dim:]

        # Using TFPL's IndependentNormal layer apparently performs worse because the scaling
        # is forced to be positive through a softplus. Using square() solves the issue.
        sigma = scale_raw ** 2  # for distribution

        res = dist.Independent(dist.Normal(mu, sigma), 1)

        # Compute KL loss if training and using exact KL
        if self.training and self.use_exact_kl:
            # Compute elementwise KL-like term
            kl_term = \
                T.log(sigma ** 2) \
                - self.kl_mu_weight * mu ** 2 \
                - sigma ** 2 \
                + 1

            kl_loss = -self.kl_weight * T.mean(T.sum(kl_term, dim=-1))
            z = res.rsample()
        else:
            kl_loss = 0
            z = res.sample()

        return z, kl_loss

    def add_kl_loss(self, samples):
        pass

    def get_config(self):
        config = super().get_config()
        # prior_loc: float = 0.0,
        # prior_scale: float = 1.0,
        # kl_weight: float = 0.5,
        # kl_mu_weight: float = 0.01,
        # use_exact_kl: bool = True,
        config.update(
            {
                "prior_loc": self.prior_loc,
                "prior_scale": self.prior_scale,
                "kl_weight": self.kl_weight,
                "kl_mu_weight": self.kl_mu_weight,
                "use_exact_kl": self.use_exact_kl,
            }
        )
        return config