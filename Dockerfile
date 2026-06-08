ARG SIMAPP_TAG=0.1-cpu
FROM awsdeepracercommunity/deepracer-env:${SIMAPP_TAG}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml /tmp/pyproject.toml
# pandas + pyarrow are required by the Tier-1 trace sink (gym_dr/trace.py). They
# usually arrive transitively via mlflow, but the sink disables itself if either
# is missing — so install them explicitly to guarantee the trace is produced
# in-container (see docs/trace-contract.md).
RUN cd /tmp && uv pip install --system --no-cache-dir \
      "stable-baselines3>=2.3" tensorboard "mlflow>=2.10" "optuna>=3.5" gymnasium \
      pandas pyarrow

ENV GYM_DR_IN_CONTAINER=1
# The project source is bind-mounted at /workspace; the container CMD runs the
# experiment file with `python3 <EXPERIMENT_PATH>`, so sys.path[0] is the
# script's own dir (e.g. /workspace/experiments) and `import gym_dr` would miss
# the package at /workspace. Put /workspace on the path so experiment files in
# any subdirectory can import gym_dr without an editable install.
ENV PYTHONPATH=/workspace

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["source /opt/ros/noetic/setup.bash && \
      source /opt/simapp/setup.bash && \
      { ./run.sh run deepracer_env.launch & } && \
      sleep 5 && \
      rosparam set WORLD_NAME ${WORLD_NAME} && \
      cd /workspace && \
      python3 -u \"${EXPERIMENT_PATH}\""]
