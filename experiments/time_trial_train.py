"""Single-run time-trial training — fixed hyperparameters, no HPO.

The "just train a car" experiment: pure time-trial driving, a fixed set of PPO
hyperparameters, and a straight ``train()`` call. No Optuna, no search space —
edit the constants below if you want, then let it run.

The hyperparameters here are lifted from the best HPO trial of the
``time_trail_min_speed_1`` study (**trial 18**, eval reward ~44k): its PPO
knobs, its custom CNN tower, its frame stack and its ``anti_zigzag`` training
reward. Two things are deliberately scaled up from that trial:

  - **10x the training time.** Trial 18 ran 250k steps on a single track. This
    run trains for ~16M steps (a 10x bump over this file's previous 1.6M
    budget) across several tracks — see the world schedule below.
  - **More PPO epochs.** ``n_epochs`` is raised from trial 18's 10 to 20, so
    each rollout batch is optimised harder before the next collection.

Beyond that it keeps the multi-track character of this experiment:

  - **Trial-18 network.** The AWS-DeepRacer-faithful ``DeepRacerCNN`` tower
    with trial 18's custom conv stack and a 256-d feature projection, feeding
    three-layer 1024-wide policy/value MLP heads.
    ``share_features_extractor=False`` gives the actor and critic their own
    independent towers; ``normalize_images=False`` feeds raw 0-255 grayscale
    (no /255), matching the physical car. See ``gym_dr/networks.py``.
  - **More tracks.** ``OrderedSplit`` trains across ``TRAIN_WORLDS`` in order,
    hot-swapping the Gazebo track between chunks via ``DeepRacerEnv.set_world``
    (one container, no per-world restart), and measures generalisation on the
    held-out ``EVAL_WORLDS`` the policy never trains on.
  - **More time.** ``CHUNK_STEPS`` per world x worlds x ``ROTATIONS`` passes —
    a much larger total budget than the single-track trial.

Run it:

    uv run python experiments/time_trial_train.py

On the host this reconstructs the experiment, pre-generates
``model_metadata.json`` and ``docker run``s a single sim container; inside the
container it trains the whole rotation in-process and writes checkpoints +
``best_model`` into a run dir under ``artifacts/``.

The GUI is on by default (``enable_gui=True``) — connect a VNC client to
``vnc://localhost:5900`` to watch the car as it trains. The sim runs at 10x
real time (``rtf_override=10``). Evaluate the result afterwards with::

    uv run python scripts/evaluate.py \\
        --model artifacts/time_trial_trial18_10x/best_model/best_model.zip

This uses GPU (``device="cuda"`` + ``use_gpu=True``) to match the built
``my-deepracer-project:gpu`` image. Switch both to CPU if you only have the
cpu image built.
"""

from gym_dr import (
    TRACKS,
    ContinuousActionSpaceConfig,
    ExperimentConfig,
    OrderedSplit,
    Sb3Trainer,
    TrackingConfig,
    TrainingConfig,
    anti_zigzag,
    progress_safe,
    time_trial,
    train,
)
from gym_dr.networks import DeepRacerCNN

# --------------------------------------------------------------------------- #
# Knobs — edit these and re-run. Everything else below is wiring.
# --------------------------------------------------------------------------- #
NAME = "tt_testing_demo_I"
FRAME_STACK = 4  # trial 18's temporal context (DeepRacerEnv emits single frames)

# --- World schedule: train on several tracks, eval generalisation on the ---
# held-out ones. Real, geometrically distinct circuits confirmed in the simapp
# image (the degenerate Oval/Straight/H shapes and reinvent_base starter track
# are excluded on purpose). Train and eval sets are disjoint so the eval reward
# measures transfer to tracks the policy never trained on.
TRAIN_WORLDS = [
    "Spain_track",  # Circuit de Barcelona-Catalunya
    "Monaco",  # European Seaside Circuit
    "Austin",  # American Hills Speedway
    "arctic_pro",  # Hot Rod Super Speedway
]
EVAL_WORLDS = [
    "reInvent2019_track",  # Smile Speedway
    "Bowtie_track",  # Bowtie Track
]
CHUNK_STEPS = 100_000  # timesteps trained per world before the track swap
ROTATIONS = 10  # full passes through TRAIN_WORLDS (2 -> 20 for the 10x budget)
# Total budget = CHUNK_STEPS x worlds x rotations = 16M steps (10x the 1.6M
# this file trained before; ~64x trial 18's single-track 250k).
TOTAL_TIMESTEPS = CHUNK_STEPS * len(TRAIN_WORLDS) * ROTATIONS

# --- Trial-18 policy/value network -----------------------------------------
# The custom conv stack Optuna built for trial 18 (base 16ch / 8-px first
# kernel, then four 32-ch refinement layers), a 256-d feature projection, then
# three-layer 1024-wide MLP heads. Hardcoded as an explicit list because it is
# not one of the named DEEPRACER_CONV_PRESETS.
CONV_LAYERS = [
    [16, 8, 4],
    [32, 4, 2],
    [32, 5, 1],
    [32, 5, 1],
    [32, 5, 1],
]  # each entry: (out_channels, kernel_size, stride)
FEATURES_DIM = 256  # width of the CNN -> MLP projection (per head)
PI_NET = [1024, 1024, 1024]  # policy (actor) MLP middleware
VF_NET = [1024, 1024, 1024]  # value (critic) MLP middleware

# Fail fast on a typo or an accidental train/eval overlap — a bad world name
# would otherwise only surface deep inside DeepRacerEnv.set_world (ValueError)
# mid-run, after the container is already up.
_unknown = sorted(set(TRAIN_WORLDS + EVAL_WORLDS) - set(TRACKS))
assert not _unknown, f"unknown world name(s) not in gym_dr.tracks.TRACKS: {_unknown}"
_overlap = sorted(set(TRAIN_WORLDS) & set(EVAL_WORLDS))
assert not _overlap, f"train/eval worlds must be disjoint; overlap: {_overlap}"


experiment = ExperimentConfig(
    name=NAME,
    env_factory=time_trial,  # pure time-trial (no object_avoidance)
    reward=anti_zigzag,  # trial 18's training reward
    eval_reward=progress_safe,  # trial 18's eval reward (also the default)
    # Fixed PPO hyperparameters — trial 18's winners, with n_epochs bumped 10->20.
    trainer=Sb3Trainer(
        name="ppo",
        policy="MultiInputPolicy",
        kwargs={
            "n_steps": 1024,
            "batch_size": 32,
            "learning_rate": 1.676613993899604e-05,
            "ent_coef": 0.05390542073198107,
            "gamma": 0.9525407607853795,
            "gae_lambda": 0.9086743940833001,
            "clip_range": 0.2015058554717573,
            "n_epochs": 20,  # trial 18 used 10 — raised for harder optimisation
            # The trial-18 AWS-faithful network: separate actor/critic CNN towers
            # (share_features_extractor=False), raw 0-255 grayscale
            # (normalize_images=False), the custom conv stack with a 256-d feature
            # projection, and three-layer 1024-wide policy/value MLP heads.
            "policy_kwargs": {
                "share_features_extractor": False,
                "normalize_images": False,
                "features_extractor_class": DeepRacerCNN,
                "features_extractor_kwargs": {
                    "conv_layers": CONV_LAYERS,
                    "features_dim": FEATURES_DIM,
                },
                "net_arch": dict(pi=PI_NET, vf=VF_NET),
            },
        },
        frame_stack=FRAME_STACK,
        device="cuda",
    ),
    action_space=ContinuousActionSpaceConfig(
        steering_low=-30.0,
        steering_high=30.0,
        # Floor speed above the crawl bound so "stay on track by barely
        # moving" isn't a viable policy — the car has to actually drive.
        speed_low=1.0,
        speed_high=4.0,
    ),
    # Multi-world train/eval split. Trains across TRAIN_WORLDS in order (hot-
    # swapping the track between CHUNK_STEPS-sized chunks) and scores the policy
    # on the held-out EVAL_WORLDS each evaluation — per-world means logged as
    # eval/<world>_mean_reward, their mean drives the best_model signal.
    world_strategy=OrderedSplit(
        train_worlds=TRAIN_WORLDS,
        eval_worlds=EVAL_WORLDS,
        chunk_steps=CHUNK_STEPS,
        rotations=ROTATIONS,
    ),
    training=TrainingConfig(
        total_timesteps=TOTAL_TIMESTEPS,
        checkpoint_freq=CHUNK_STEPS,
        # Keep only the most recent few checkpoints. At ~105 MB each (big net) a
        # 16M-step run at this freq would otherwise hoard ~160 zips (~16 GB) and
        # fill the disk. best/final/latest_model live outside checkpoints/.
        checkpoint_keep_last=3,
        # Each eval rolls out n_eval_episodes on EVERY held-out world, so eval
        # cost scales with len(EVAL_WORLDS). 200k keeps the eval count sane over
        # the much larger 16M budget.
        eval_freq=CHUNK_STEPS,
        n_eval_episodes=3,
        rtf_override=160,  # run the sim at 10x real time
        # Render the driven trajectory over a skeleton of each eval track to
        # TensorBoard's Images tab: one overlay per eval world (all 3 episodes,
        # colour + legend) + one chart per episode. See gym_dr/trainers/sb3/plots.py.
        eval_path_plots=True,
    ),
    tracking=TrackingConfig(mlflow_experiment=NAME),
    # Watch the car train over VNC: connect a client to vnc://localhost:5900.
    enable_gui=True,
    seed=42,
    use_gpu=True,
)


if __name__ == "__main__":
    train(experiment)
