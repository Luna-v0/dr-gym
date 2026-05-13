ARG SIMAPP_TAG=0.1-cpu
FROM awsdeepracercommunity/deepracer-env:${SIMAPP_TAG}

# Bake only third-party deps into the image. The project source (train.py,
# reward.py, model_metadata.json, configs) is bind-mounted at /workspace at
# run time, so editing those files does NOT require rebuilding the image.
# Rebuild is only needed when requirements.txt changes.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["source /opt/ros/noetic/setup.bash && source /opt/simapp/setup.bash && { ./run.sh run deepracer_env.launch & } && sleep 5 && rosparam set WORLD_NAME ${WORLD_NAME} && python3 -u /workspace/train.py"]
