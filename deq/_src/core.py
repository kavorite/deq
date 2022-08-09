from typing import NamedTuple, Optional

import diffrax as dax
import haiku as hk
import jax
import numpy as np
from reparam import FiLM

from .util import pos_emb


class DEQSolver(NamedTuple):
    wrap: dax.AbstractSolver = dax.Tsit5()
    rtol: float = 5e-3
    atol: float = 1e-4
    max_depth: Optional[int] = None

    def __call__(self, apply_fn, params, states):
        @dax.ODETerm
        def ode_term(depth, states, params):
            return apply_fn(params, depth, states) - states

        solution = dax.diffeqsolve(
            ode_term,
            self.wrap,
            t0=0,
            t1=float("inf"),
            dt0=None,
            y0=states,
            args=params,
            max_steps=self.max_depth,
            stepsize_controller=dax.PIDController(rtol=self.rtol, atol=self.atol),
            adjoint=dax.ImplicitAdjoint(),
            discrete_terminating_event=dax.SteadyStateEvent(rtol=self.rtol),
        )
        return solution.ys[0]


class DEQCell(hk.Module):
    def __init__(self, cell, max_depth=None, activation=jax.nn.silu, name=None):
        super().__init__(name=name)
        self.cell = cell
        self.film = FiLM()
        self.actn = activation
        self.max_depth = max_depth

    def __call__(self, depth, states):
        d = self.max_depth or states.shape[-1]
        depth_embedding = pos_emb(width=d, period=d, depth=depth) / np.sqrt(d)
        return self.actn(states + self.cell(self.film(states, depth_embedding)))


class DEQCore(hk.Module):
    def __init__(self, cell, resolver=DEQSolver(), name=None):
        super().__init__(name=name)
        cell = DEQCell(cell) if not isinstance(cell, DEQCell) else cell
        self.cell = hk.transform(cell)
        self.resolver = resolver

    @hk.transparent
    def __call__(self, inputs):
        params = hk.lift(self.cell.init, name="deq_cell")(
            hk.next_rng_key() if hk.running_init() else None, depth=0, states=inputs
        )
        rng = hk.maybe_next_rng_key()
        z_init = self.cell.apply(params, rng, depth=0, states=inputs)

        def apply_fn(params, depth, states):
            return self.cell.apply(params, rng, depth, states)

        z_star = self.resolver(apply_fn, params, z_init)
        return z_init, z_star
