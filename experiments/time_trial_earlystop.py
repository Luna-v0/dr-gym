"""Curriculum time-trial with track-mastery early stop.

A variant of ``time_trial_train.py`` that demonstrates the **early-stop on track
mastery** heuristic (``TrainingConfig.early_stop_*``). Instead of training a
fixed budget per track, the car *moves on as soon as it can drive a track
without leaving it*:

  - It trains track-by-track through ``CURRICULUM`` (easier circuits first).
  - At each evaluation the trainer measures how many eval episodes ended with
    the car off the track. Once that rate is within ``MAX_OFFTRACK_RATE`` for
    ``PATIENCE`` consecutive evals, the chunk ends early and the rotation hot-
    swaps to the **next** track (``DeepRacerEnv.set_world``, one container).
  - ``CHUNK_STEPS`` is now an *upper bound* per track: a track the car never
    masters still hands off after ``CHUNK_STEPS`` so the curriculum keeps moving.
  - On the **last** track of the **last** rotation, mastering it simply ends the
    run (status ``early_stopped``) instead of advancing.

Because mastery is judged on the *current training track*, this uses
``FixedWorlds`` (eval on the world being trained) rather than the
``OrderedSplit`` held-out split — "stay on this track ⇒ advance" is exactly the
signal we want. Set ``MAX_OFFTRACK_RATE = 0.0`` (the default) to demand the car
never leave the track across the eval episodes; raise it (e.g. ``0.5``) to
advance once it leaves on at most half of them.

Everything else — the trial-18 PPO knobs, the AWS-faithful ``DeepRacerCNN``
tower, the frame stack and the ``anti_zigzag`` reward — is carried over verbatim
from ``time_trial_train.py``.

Run it:

    uv run python experiments/time_trial_earlystop.py

This uses GPU (``device="cuda"`` + ``use_gpu=True``) to match the built
``my-deepracer-project:gpu`` image. Switch both to CPU if you only have the cpu
image built. Evaluate the result afterwards with::

    uv run python scripts/evaluate.py \\
        --model artifacts/tt_curriculum_earlystop/best_model/best_model.zip
"""

from gym_dr import (
    TRACKS,
    ContinuousActionSpaceConfig,
    ExperimentConfig,
    Sb3Trainer,
    FixedWorlds,
    TrackingConfig,
    TrainingConfig,
    OfftrackRate,
    anti_zigzag,
    progress_safe,
    time_trial,
    train,
)
from gym_dr.networks import DeepRacerCNN

# --------------------------------------------------------------------------- #
# Knobs — edit these and re-run. Everything else below is wiring.
# --------------------------------------------------------------------------- #
NAME = "tt_curriculum_earlystop"
FRAME_STACK = 4  # trial 18's temporal context (DeepRacerEnv emits single frames)

# --- Curriculum: train these tracks in order, advancing as each is mastered. --
# Roughly easier -> harder so the early-stop heuristic has a chance to bank a
# track and move on before the per-track CHUNK_STEPS cap. Real, geometrically
# distinct circuits confirmed in the simapp image.
CURRICULUM = [
    "Bowtie_track",  # Bowtie Track — gentle
    "reInvent2019_track",  # Smile Speedway
    "Spain_track",  # Circuit de Barcelona-Catalunya
    "Monaco",  # European Seaside Circuit
    "Austin",  # American Hills Speedway
    "arctic_pro",  # Hot Rod Super Speedway — hard
]
CHUNK_STEPS = 100_000  # UPPER BOUND of timesteps per track before a forced swap
ROTATIONS = 10  # passes through CURRICULUM (re-visits re-confirm mastery fast)
# Worst-case budget if NOTHING is ever mastered (early stop only ever shortens
# this): CHUNK_STEPS x tracks x rotations.
MAX_TIMESTEPS = CHUNK_STEPS * len(CURRICULUM) * ROTATIONS

# --- Early-stop on track mastery -------------------------------------------- #
# Advance to the next track once <= MAX_OFFTRACK_RATE of an eval round's episodes
# ended off-track, held for PATIENCE consecutive evals. 0.0 = the car must finish
# every eval episode without leaving the track.
MAX_OFFTRACK_RATE = 0.0
PATIENCE = 2  # consecutive clean evals required (guards against a lucky eval)

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

# Fail fast on a typo — a bad world name would otherwise only surface deep inside
# DeepRacerEnv.set_world (ValueError) mid-run, after the container is already up.
_unknown = sorted(set(CURRICULUM) - set(TRACKS))
assert not _unknown, f"unknown world name(s) not in gym_dr.tracks.TRACKS: {_unknown}"


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
    # Curriculum rotation: train CURRICULUM in order, hot-swapping the Gazebo
    # track between chunks. With no held-out eval set, each eval scores the
    # CURRENT training world, so the early-stop heuristic advances the moment the
    # car masters the track it is on.
    world_strategy=FixedWorlds(
        names=CURRICULUM,
        chunk_steps=CHUNK_STEPS,
        rotations=ROTATIONS,
    ),
    training=TrainingConfig(
        total_timesteps=MAX_TIMESTEPS,
        checkpoint_freq=CHUNK_STEPS,
        # Keep only the most recent few checkpoints (each ~105 MB for this net).
        # best/final/latest_model live outside checkpoints/ and are never pruned.
        checkpoint_keep_last=3,
        # Evaluate often enough that a mastered track is detected well before the
        # CHUNK_STEPS cap — the eval off-track rate is what triggers the advance.
        eval_freq=20_000,
        n_eval_episodes=3,
        rtf_override=160,  # run the sim at 10x real time
        # Render the driven trajectory over a skeleton of the eval track to
        # TensorBoard's Images tab. See gym_dr/trainers/sb3/plots.py.
        eval_path_plots=True,
        # --- The early-stop heuristic this experiment demonstrates ---
        early_stop=OfftrackRate(max_offtrack_rate=MAX_OFFTRACK_RATE, patience=PATIENCE),
    ),
    tracking=TrackingConfig(mlflow_experiment=NAME),
    # Watch the car train over VNC: connect a client to vnc://localhost:5900.
    #enable_gui=True,
    seed=42,
    use_gpu=True,
)


if __name__ == "__main__":
    train(experiment)
