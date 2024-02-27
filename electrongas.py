import netket as nk
import jax.numpy as jnp

from src._potentials import ewald_coulomb
from src.MPNQS import MPNQS


def mycb(step, logged_data, driver):
    logged_data["acceptance"] = float(driver.state.sampler_state.acceptance)
    return True


N = 14
# unpolarized
n_per_spin = (N // 2, N // 2)
sdim = 3
L = (4 * jnp.pi / 3 * N) ** (1 / 3)
rs = 2.0

hilb = nk.hilbert.Particle(N=N, L=(L, L, L), pbc=True)
sa = nk.sampler.MetropolisGaussian(hilb, sigma=0.04, n_chains=12, n_sweeps=64)

potential = lambda x: ewald_coulomb(x, L, alpha=1.0, kmax=20)
epot = 1 / rs * nk.operator.PotentialEnergy(hilb, potential)
ekin = nk.operator.KineticEnergy(hilb, mass=rs**2)
ha = ekin + epot

model = MPNQS(n_per_spin=n_per_spin, sdim=3, L=L)
vs = nk.vqs.MCState(sa, model, n_samples=1024 * 1, n_discard_per_chain=32)
vs.chunk_size = 1

op = nk.optimizer.Sgd(0.05)
sr = nk.optimizer.SR(diag_shift=0.0001)

log = nk.logging.JsonLog("HEG_14particles_test", save_params_every=7)

gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)
gs.run(n_iter=150, callback=mycb, out=log)
