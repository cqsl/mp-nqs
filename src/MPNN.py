import jax.numpy as jnp
import numpy as np

import flax.linen as nn
from jax.nn.initializers import lecun_normal

from src.mlp import MLP

from typing import Callable


class Networks:
    def __init__(
        self, intermediate_dim, num_intermediate, embedding_dim, activation=nn.gelu
    ):
        """Create networks for the MPNQS
        Args:
            intermediate_dim: dimension of the intermediate coordinates
            num_intermediate: number of hidden layers
            embedding_dim: dimension of the final coordinates of the layer
        """
        self.intermediate_dim = intermediate_dim
        self.num_intermediate = num_intermediate
        self.embedding_dim = embedding_dim
        self.activation = activation

    def Net(self):
        return MLP(
            self.embedding_dim,
            self.num_intermediate * (self.intermediate_dim,),
            activation=self.activation,
        )


class GraphUpdate(nn.Module):
    """Upplies the Particle Attention Graph Updates"""

    networks: Callable
    L: float
    n_up: int
    n_down: int
    attention_dim: int

    @nn.compact
    def __call__(self, x, xij, x0, xij0, i, final):
        """
        x: Current nodes of the graph
        xij: Current edges of the graph
        x0: Initial nodes of the graph
        xij0: Initial edges of the graph
        i: iteration
        final: bool whether it is the last iteration
        """

        # Networks
        phi = self.networks.Net()
        f = self.networks.Net()
        f2 = self.networks.Net()

        M, N, _ = x.shape

        wquery = self.param(
            f"query_{i}",
            lecun_normal(),
            (xij.shape[-1], self.attention_dim),
            np.float64,
        )
        wkey = self.param(
            f"key_{i}", lecun_normal(), (xij.shape[-1], self.attention_dim), np.float64
        )
        xquery = jnp.dot(xij, wquery)
        xkey = jnp.dot(xij, wkey)

        weight = f2(jnp.einsum("...ijk,...jlk->...ilk", xquery, xkey) / jnp.sqrt(32))

        message = phi(xij) * weight

        xnew = jnp.concatenate(
            (x0, f(jnp.concatenate((x, jnp.sum(message, axis=-2)), axis=-1))), axis=-1
        )
        if not final:
            g = self.networks.Net()
            xij = jnp.concatenate(
                (xij0, g(jnp.concatenate((xij, message), axis=-1))), axis=-1
            )
        return xnew, xij


class MPNN(nn.Module):
    n_up: int
    n_down: int
    embedding_dim: int
    intermediate_dim: int
    num_intermediate: int
    attention_dim: int
    L: float = 1.0
    n_interactions: int = 1

    @nn.compact
    def __call__(self, x):
        assert len(x.shape) == 3
        M, N, sdim = x.shape

        # network generator
        networks = Networks
        network_factory = networks(
            self.intermediate_dim, self.num_intermediate, self.embedding_dim
        )

        # embeddings
        X = nn.Embed(
            num_embeddings=3,
            features=self.embedding_dim,
            dtype=np.float64,
            param_dtype=np.float64,
            embedding_init=lecun_normal(),
        )
        xembed = jnp.tile(X(jnp.array([0, 1, 2]))[None, ...], (M, 1, 1))

        # spin products (assuming first coordinates are spin up and then spin down)
        s = jnp.hstack((self.n_up * [1], self.n_down * [-1]))
        ss = jnp.outer(s, s)[..., None]

        # compute periodic distances and their norm
        dist = x[..., :, None, :] - x[..., None, :, :]
        d = jnp.linalg.norm(
            jnp.sin(jnp.pi / self.L * dist) + jnp.eye(N)[..., None],
            axis=-1,
            keepdims=True,
        ) * (1.0 - jnp.eye(N)[..., None])
        distp = jnp.concatenate(
            (jnp.sin(2 * jnp.pi / self.L * dist), jnp.cos(2 * jnp.pi / self.L * dist)),
            axis=-1,
        )

        # nodes (translation invariance requires no single particle coordinates)
        x0 = jnp.tile(xembed[:, 0, :][:, None, :], (1, N, 1))
        x = jnp.concatenate(
            (x0, jnp.tile(xembed[:, 1, :][:, None, :], (1, N, 1))), axis=-1
        )

        # edges
        xij0 = jnp.concatenate(
            (distp, d, jnp.tile(ss[None, :, :, :], (x.shape[0], 1, 1, 1))), axis=-1
        )
        xij = jnp.concatenate(
            (xij0, jnp.tile(xembed[:, 2, :][:, None, None, :], (1, N, N, 1))), axis=-1
        )

        for i in range(self.n_interactions):
            layer = GraphUpdate(
                network_factory, self.L, self.n_up, self.n_down, self.attention_dim
            )

            x, xij = layer(x, xij, x0, xij0, i, i == self.n_interactions - 1)

        return x
