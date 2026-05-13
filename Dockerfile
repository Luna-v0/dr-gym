ARG SIMAPP_TAG=0.1-cpu
FROM awsdeepracercommunity/deepracer-env:${SIMAPP_TAG}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml /tmp/pyproject.toml
RUN cd /tmp && uv pip install --system --no-cache-dir \
      "stable-baselines3>=2.3" tensorboard "mlflow>=2.10" "optuna>=3.5" gymnasium

ENV GYM_DR_IN_CONTAINER=1

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["source /opt/ros/noetic/setup.bash && \
      source /opt/simapp/setup.bash && \
      { ./run.sh run deepracer_env.launch & } && \
      sleep 5 && \
      rosparam set WORLD_NAME ${WORLD_NAME} && \
      cd /workspace && \
      python3 -u \"${EXPERIMENT_PATH}\""]
