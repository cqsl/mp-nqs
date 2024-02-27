import jax.numpy as jnp
import jax

from flax import linen as nn
from jax.nn.initializers import lecun_normal

from netket.utils import HashableArray
from netket.utils.types import DType, NNInitFunc


def diagonal_init(l, idx, dtype=float):
    def init(key, shape, dtype=dtype):
        eye = jnp.eye(l)
        noise = jax.random.normal(key, eye.shape)
        return eye + noise

    return init


class PW(nn.Module):
    """create plane wave orbitals"""
    kvecs: HashableArray  # rows are k's
    N_orbs: int
    combine: bool = False
    param_init: NNInitFunc = lecun_normal()
    dtype: DType = float

    @nn.compact
    def __call__(self, x):
        """Plane wave orbitals
        x: (..., N, D)
        """

        # ikx #(...,N, N_k)
        res = 1j * jnp.einsum("ij,...j->...i", self.kvecs, x)

        if self.combine:
            params = self.param('PlaneWaveCoefficients', self.param_init, (res.shape[-1], self.N_orbs), self.dtype)
            res = jnp.einsum('ij,...i->...j', params, res)

        return res
