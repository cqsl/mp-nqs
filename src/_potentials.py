import jax
import jax.numpy as jnp
import numpy as np

from functools import partial


@partial(jax.jit, static_argnums=(1,))
def smallest_vecs(basis, n):
    dim = basis.shape[-1]
    r = np.arange(-n, (n + 1))  # produces O(n*n*n) numbers

    vecs = jnp.array(np.meshgrid(*(dim * (r,)))).T.reshape(-1, dim)
    return vecs


def ewald_coulomb(x, L, alpha, sdim=3, kmax=20, cutoff=None):
    """Ewald summation for cubic lattices."""

    if sdim != 3:
        raise ValueError("Ewald summation only implemented for 3 spatial dimensions.")

    if cutoff is None:
        cutoff = 2.0 * L

    N = x.shape[0] // sdim
    x = x.reshape(-1, N, sdim)

    dists = x[..., :, None, :] - x[..., None, :, :]
    V = L**sdim

    def real_sum(dists):
        """Real space part of Ewald summation"""

        # vec(R)=0
        mask = 1 - jnp.eye(N)
        d = jnp.linalg.norm(dists, axis=-1)
        res1 = jnp.where(
            mask[jnp.newaxis, :], jax.scipy.special.erfc(alpha * d) / d, 0
        ).reshape(-1)

        # vec(R)=L * vec(n)
        Rn = smallest_vecs(jnp.eye(sdim), 2) * L
        temp = jnp.linalg.norm(Rn, axis=-1)
        Rn = jnp.where(temp[:, None] <= cutoff, Rn, 0)  # exlude too far away neighbours
        temp = jnp.linalg.norm(Rn, axis=-1)
        d = jnp.linalg.norm(dists[..., None, :] + Rn, axis=-1)

        res2 = jnp.where(temp == 0, 0, jax.scipy.special.erfc(alpha * d) / d).reshape(
            -1
        )  # Rn=0 is special

        res = jnp.sum(res1) + jnp.sum(res2)
        return res

    def recip_sum(x):
        """Reciprocal space part of Ewald summation"""

        # vec(k)=1/L * vec(n)
        Gm = smallest_vecs(jnp.eye(sdim), kmax) / L
        # |vec(k)|>kmax
        G = jnp.linalg.norm(Gm, axis=-1, keepdims=True)
        Gm = jnp.where(G <= kmax / L, Gm, 0)
        G = jnp.linalg.norm(Gm, axis=-1)
        G2 = (G**2).reshape(-1)

        t1 = jnp.where(G2 == 0, 0, jnp.exp(-G2 * jnp.pi**2 / (alpha**2)) / G2)
        # structure factor
        S = jnp.sum(
            jnp.where(
                G2 == 0,
                0,
                jnp.exp(1j * 2 * jnp.pi * jnp.einsum("...ij,kj->...ik", x, Gm)),
            ),
            axis=-2,
        ).reshape(-1)

        t2 = jnp.abs(S) ** 2

        res = 1 / (V * jnp.pi) * jnp.sum(t1 * t2)
        return res

    def self_energy():
        return -N * alpha / jnp.sqrt(jnp.pi)

    def constant():
        V = L**sdim
        return -(N**2) * jnp.pi / (2 * alpha**2 * V)

    ereal = real_sum(dists)
    erec = recip_sum(x)
    eself = self_energy()
    const = constant()

    return 0.5 * ereal + 0.5 * erec + eself + const
