import flax.linen as nn

import numpy as np
from typing import Any, Callable, Tuple


class MLP(nn.Module):
    out_dim: int
    hidden_layers: Tuple[int]
    activation: Callable = nn.gelu
    output_activation: Callable = None
    last_bias: bool = True
    last_linear: bool = True
    auto_width: bool = False
    dtype: Any = float
    kwargs: dict = None

    @nn.compact
    def __call__(self, x):
        in_dim = x.shape[-1]
        kwargs = self.kwargs if self.kwargs is not None else {}

        if self.auto_width and len(self.hidden_layers) == 1:
            n_hidden = self.hidden_layers[0]
            qs = [k / n_hidden for k in range(n_hidden + 1)]
            dims = [int(np.round(in_dim ** (1 - q) * self.out_dim**q)) for q in qs]
        else:
            dims = [in_dim, *self.hidden_layers, self.out_dim]

        for k in range(len(dims) - 1):
            last = k + 2 == len(dims)
            bias = not last or self.last_bias or not self.last_linear

            x = nn.Dense(
                dims[k + 1],
                use_bias=bias,
                param_dtype=self.dtype,
                name=f"linear{k+1}",
                **kwargs,
            )(x)

            if self.output_activation is not None:
                x = self.output_activation(x)

            if not last or not self.last_linear and not self.output_activation:
                x = self.activation(x)

        return x
