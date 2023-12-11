import jax.numpy as jnp

from flax import linen as nn

from netket.utils import HashableArray
from netket.utils.types import DType


def diagonal_init(l, idx, dtype=float):
    def init(key, shape, dtype=dtype):
        eye = jnp.eye(l)
        return eye

    return init


class PW(nn.Module):
    """create plane wave orbitals"""

    kvecs: HashableArray  # rows are k's
    dtype: DType = float

    @nn.compact
    def __call__(self, x):
        """Plane wave orbitals
        x: (..., N, D)
        """

        # ikx #(...,N, N_k)
        res = 1j * jnp.einsum("ij,...j->...i", self.kvecs, x)
        return res
