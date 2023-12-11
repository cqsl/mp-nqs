import numpy as np
from jax import numpy as jnp

from flax import linen as nn
from jax.nn.initializers import ones, lecun_normal, variance_scaling
from src.mlp import MLP

from netket.utils.types import DType, NNInitFunc
from typing import Tuple

from src.slater import LogSlaterDet
from src.pw import PW
from src.MPNN import MPNN


def smallest_kvecs(basis, m, n):
    dim = basis.shape[-1]
    r = np.arange(-n, (n + 1))  # produces O(n*n*n) numbers

    vecs = np.array(np.meshgrid(*(dim * (r,)))).T.reshape(-1, dim)
    vecs = vecs[np.argsort(np.linalg.norm(vecs, axis=1))]
    return vecs[:m, :]


class MPNQS(nn.Module):
    n_per_spin: Tuple[int]
    sdim: int = 1
    L: float = 1.0
    paramsre_init: NNInitFunc = lecun_normal()
    params_init: NNInitFunc = variance_scaling(
        0.0001, "fan_in", "truncated_normal"
    )  # lecun_normal()
    scale_init: NNInitFunc = ones
    dtype: DType = jnp.float64

    def setup(self):
        """Construction of plane wave orbitals"""
        k_vec = (
            2
            * np.pi
            / self.L
            * jnp.concatenate(
                (
                    smallest_kvecs(jnp.eye(self.sdim), self.n_per_spin[0], 10),
                    smallest_kvecs(jnp.eye(self.sdim), self.n_per_spin[1], 10),
                ),
                axis=0,
            )
        )
        self.k_vec = k_vec
        pw_basis = {}

        pw_basis["orbtials_up"] = PW(k_vec[: self.n_per_spin[0]])
        pw_basis["orbtials_down"] = PW(k_vec[self.n_per_spin[0] :])
        self.orb_basis = pw_basis

        self.MPNN = MPNN(
            *self.n_per_spin,
            embedding_dim=32,
            intermediate_dim=32,
            num_intermediate=2,
            attention_dim=32,
            L=self.L,
            n_interactions=1
        )
        self.slater = LogSlaterDet(n_per_spin=self.n_per_spin, orbitals=self.orb_basis)
        self.net = MLP(out_dim=1, hidden_layers=(32,))

    @nn.compact
    def __call__(self, x):
        """input: x = (..., N*d)"""

        sha = x.shape
        N = sum(self.n_per_spin)
        x = x.reshape(-1, sum(self.n_per_spin), self.sdim)
        M = x.shape[0]

        # backflow
        y = self.MPNN(x)
        prealup = self.param(
            "realup", self.paramsre_init, (y.shape[-1], self.sdim), self.dtype
        )
        pimagup = self.param(
            "imagup", self.params_init, (y.shape[-1], self.sdim), self.dtype
        )
        bf = x + jnp.einsum("kd, ...ik->...id", prealup + 1j * pimagup, y)

        k_vec = jnp.tile(self.k_vec[None, None, :, :], (M, N, 1, 1))

        jastrow = jnp.tile(y[:, :, None, :], (1, 1, N, 1))
        jastrow = jnp.concatenate((jastrow, k_vec), axis=-1)
        jastrow = self.net(jastrow).squeeze(axis=-1)
        psi = self.slater(bf, jastrow)

        return psi.reshape(sha[0])
