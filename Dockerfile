ARG SIMAPP_TAG=0.1-cpu
FROM awsdeepracercommunity/deepracer-env:${SIMAPP_TAG}

# Install third-party dependencies first — cached unless requirements.txt changes
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Copy and install your project
COPY . /workspace
RUN pip install --no-cache-dir -e /workspace

# Default: run your training script
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["source /opt/ros/noetic/setup.bash && source /opt/simapp/setup.bash && { ./run.sh run deepracer_env.launch & } && sleep 5 && rosparam set WORLD_NAME ${WORLD_NAME} && python3 -u /workspace/train.py"]
