import flax.linen as nn
import scipy
import numpy as np
from typing import Any, Callable, Sequence, Tuple
import jax.numpy as jnp


def slogdet(x):
    sign, logdet = jnp.linalg.slogdet(x)
    return sign, logdet


class LogSlaterDet(nn.Module):
    n_per_spin: Tuple[int]
    orbitals: Sequence[Callable[..., Any]]
    return_signs: bool = False
    add_signs: bool = True

    @nn.compact
    def __call__(self, x, orbs=None, mask=True):
        """
        x: (..., N, sdim) Particle positions on which to evaluate the orbtials.
        orbs: (...,N_orbs) Orbital aware Jastrow factor
        mask: whether or not to make the Slater determinant block diagonal
        """

        if not self.orbitals:
            raise ValueError(f"Empty LogSlaterDet module {self.name}.")

        assert (
            x.ndim == 3
        ), f"Got shape {x.shape} where (..., N_particles, sdim) was expected."
        N = x.shape[-2]

        mat = []
        for key, value in self.orbitals.items():
            o = value(x)
            mat.append(o)

        mat = jnp.concatenate(mat, axis=-1)

        assert mat.shape[-2] == mat.shape[-1], (
            f"The matrix constructed from the orbtials has shape {mat.shape} and is "
            f"non-square."
        )

        if mask:
            """mask to make sure that off diagonal blocks are zero"""
            x = jnp.tile(
                jnp.array(self.n_per_spin[0] * [1] + self.n_per_spin[1] * [0])[:, None],
                (1, self.n_per_spin[0]),
            )
            y = jnp.tile(
                jnp.array(self.n_per_spin[0] * [0] + self.n_per_spin[1] * [1])[:, None],
                (1, self.n_per_spin[1]),
            )
            mask = jnp.concatenate((x, y), axis=-1)

            # logdet works in normal space
            mat = mask[..., :, :] * jnp.exp(mat)

        if orbs is not None:
            mat = mat * jnp.exp(orbs)
        assert mat.shape[-1] == mat.shape[-2], (
            f"The matrix constructed from the orbtials has shape {mat.shape} and is "
            f"non-square."
        )

        signs, logslaterdet = jnp.linalg.slogdet(mat)
        # normalization
        log_norm = 0.5 * np.log(scipy.special.factorial(N))
        logslaterdet = logslaterdet - log_norm

        if self.add_signs:
            logslaterdet = logslaterdet + jnp.log(signs)

        if self.return_signs:
            return signs, logslaterdet
        else:
            return logslaterdet
