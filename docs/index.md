# dr-gym

A clean, extensible, **reproducible** orchestrator for reinforcement-learning
studies on Gymnasium environments — built for DeepRacer, but algorithm- and
environment-agnostic. You define a study in config, run it, and analyse the
results; new algorithms and environments plug in by implementing a small,
documented interface.

## Start here

- **[The Study / Pipeline interface](study-pipeline.md)** — how to run an
  experiment: the single `Study` entrypoint (training *and* hyperparameter search),
  the `Trainer` abstract class (bring your own algorithm), the composable `Stage`
  pipeline, early-stopping strategies, and curriculum.
- **[Configuration](configuration.md)** — the typed `EnvironmentConfig` /
  `ExperimentConfig` authoring surface.
- **[Algorithms](algorithms.md)** and **[Hyperparameter optimization](hpo.md)**.

## How it fits together

- **[System overview](system-overview.md)** · **[Code map](code-map.md)**
- **[Trainer contract](trainer-contract.md)** — the `Trainer` / `TrainingContext`
  services every algorithm gets (TensorBoard + MLflow, checkpoints, eval, pruning).
- **[Trace contract](trace-contract.md)** · **[Artifact layout](artifact-layout.md)**

## Evaluate & observe

- **[Evaluation protocol](eval-protocol.md)** — clean-completion as the headline metric.
- **[Tracking (MLflow)](tracking.md)** · **[TensorBoard](tensorboard.md)**

## Deploy

- **[ONNX support](onnx-support-status.md)** · **[Physical car](physical-car-integration-notes.md)**

The simulation layer (ROS 2 Lyrical / Gazebo Jetty) is documented separately in the
**deepracer-env** repository.
