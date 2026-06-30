# Base = the deepracer-env simulator image. As of the ROS 2 Lyrical / Gazebo
# Jetty port this is the ament/colcon-built Lyrical runtime image; bootstrap.sh
# builds and SHA-pins it from the deepracer-env port branch. Same tag scheme as
# before (0.1-<arch>[-<sha>]) so the versioning logic is unchanged.
ARG SIMAPP_TAG=0.1-cpu
FROM awsdeepracercommunity/deepracer-env:${SIMAPP_TAG}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml /tmp/pyproject.toml
# pandas + pyarrow are required by the Tier-1 trace sink (gym_dr/trace.py). They
# usually arrive transitively via mlflow, but the sink disables itself if either
# is missing — so install them explicitly to guarantee the trace is produced
# in-container (see docs/trace-contract.md).
# The Lyrical base's system Python (3.14) is PEP-668 externally-managed, so
# --break-system-packages is required. Install CPU-only torch FIRST from the
# pytorch CPU index: the default index resolves torch's full CUDA 13 stack
# (multi-GB, unwanted on the CPU/laptop box). SB3 then sees torch satisfied and
# does not pull CUDA. (The GPU box overrides TORCH_INDEX for a CUDA build.)
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN cd /tmp && uv pip install --system --break-system-packages --no-cache-dir \
      --index-url ${TORCH_INDEX} torch \
 && uv pip install --system --break-system-packages --no-cache-dir \
      "stable-baselines3>=2.3" tensorboard "mlflow>=2.10" "optuna>=3.5" gymnasium \
      pandas pyarrow

ENV GYM_DR_IN_CONTAINER=1
# Modern mlflow (3.x) raises on the './mlruns' file-store backend ("maintenance
# mode") unless opted in. dr-gym uses the file store (mlflow_tracking_uri
# file:./mlruns) and the host analysis pipeline reads that tree, so keep it.
ENV MLFLOW_ALLOW_FILE_STORE=true
# The project source is bind-mounted at /workspace; the container CMD runs the
# experiment file with `python3 <EXPERIMENT_PATH>`, so sys.path[0] is the
# script's own dir (e.g. /workspace/experiments) and `import gym_dr` would miss
# the package at /workspace. Put /workspace on the path so experiment files in
# any subdirectory can import gym_dr without an editable install.
ENV PYTHONPATH=/workspace

ENTRYPOINT ["/bin/bash", "-c"]
# ROS 2 Lyrical changes vs the Noetic CMD:
#   * source /opt/ros/lyrical (was noetic); /opt/simapp overlay is unchanged.
#   * WORLD_NAME is passed straight through as an environment variable — ROS 2
#     has no global `rosparam` server, and the Jetty launch reads $WORLD_NAME at
#     startup (then DeepRacerEnv.set_world() swaps tracks between chunks). So the
#     `rosparam set` line is gone; the env var already crosses the boundary.
#   * `ros2 launch ... deepracer_env.launch.py` replaces `roslaunch`.
CMD ["source /opt/ros/lyrical/setup.bash && \
      source /opt/simapp/setup.bash && \
      { ros2 launch deepracer_simulation_environment deepracer_env.launch.py & } && \
      sleep 8 && \
      cd /workspace && \
      python3 -u \"${EXPERIMENT_PATH}\""]
