"""Throwaway: reproduce ONE LSTM trial through the full train() pipeline to surface
the real traceback the HPO study swallows. Tiny budget, no eval, first train track."""
import os
os.environ["GYM_DR_FEATURE_SET"] = "actor_extended"

import dataclasses                                           # noqa: E402
try:                                                         # noqa: E402
    import experiments.oracle_hpo as h
except ModuleNotFoundError:
    import oracle_hpo as h
from gym_dr import Study                                     # noqa: E402


class _FT:
    def __init__(self, a): self.a = a
    def suggest_categorical(self, n, c): return self.a if n == "arch" else c[0]
    def suggest_float(self, n, a, b, log=False): return (a + b) / 2


_cfg = h.base.with_overrides(name="arch_lstm_repro", **h.search_space(_FT("lstm")))
_cfg = dataclasses.replace(
    _cfg,
    training=dataclasses.replace(_cfg.training, total_timesteps=2000, eval_freq=10 ** 9,
                                 checkpoint_freq=10 ** 9),
    environment=dataclasses.replace(
        _cfg.environment,
        curriculum=dataclasses.replace(_cfg.environment.curriculum, n_chunks=1,
                                       chunk_steps=2000, eval_worlds=[])),
)
experiment = _cfg

if __name__ == "__main__":
    Study(experiment).run()