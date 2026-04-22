# Physical Car Integration Notes

Date: 2026-04-08

## Current repo status

This repo currently works as a local CPU-first training setup built on top of `deepracer-env`.

What is already working here:

- Local training can run inside the simulator on CPU.
- Checkpoints and final models are persisted to host-mounted `artifacts/`.
- The training flow writes:
  - `final_model.zip`
  - `latest_model.zip`
  - periodic checkpoints
  - `model_metadata.json`
  - `reward_function.py`
  - `training_status.json`

Important limitation:

- The saved `stable-baselines3` `.zip` files are resumable for local training.
- They are **not** the same as an AWS DeepRacer-native model bundle used by the stock AWS model import/export flow.

## What this stack does not currently provide

The local `deepracer-env + stable-baselines3 + PyTorch` stack does **not** include a supported exporter from SB3/PyTorch checkpoints to the native AWS DeepRacer physical-car model format.

In practice, this means:

- local training works
- local checkpointing works
- local resume works
- stock AWS physical-car import/export compatibility does **not** come for free

## Why the stock AWS model path is a poor target

The AWS open-source device stack still assumes a model-management path centered around:

- `model_loader_node`
- `model_optimizer_node`
- `inference_node`

The open-source `model_loader_node` is built around AWS model artifacts under `/opt/aws/deepracer/artifacts` and handles files such as:

- `.pb`
- `.json`
- `.tar`
- `.gz`

The stock AWS autonomous path also expects model optimization through Intel OpenVINO before inference.

This is the key technical mismatch:

- our local trainer produces SB3/PyTorch checkpoints
- the AWS autonomous model path is built for AWS-native/OpenVINO-oriented model artifacts

Trying to make SB3/PyTorch look like a stock AWS DeepRacer model would likely require reverse-engineering and replacing multiple assumptions at once:

- model bundle format
- model loading behavior
- optimizer inputs
- inference runtime expectations
- frontend/backend model management expectations

That is the high-risk path.

## Recommended path

The realistic path is:

- keep the AWS OS
- keep the AWS ROS stack
- keep the AWS camera, servo, manual mode, calibration, and system services
- add a **parallel custom mode** for our own inference

This is the same overall pattern AWS used for the `Follow the Leader` sample project: they did **not** replace the whole device software, they extended the control plane and added a new driving mode.

Recommended integration boundary:

- do **not** patch stock `autonomous` mode to accept SB3/PyTorch
- do **not** patch stock `model_loader_node` to fake AWS-native bundles
- do add a new custom mode, e.g. `customrl`

## What can be reused from AWS

The existing AWS stack already gives us most of the machinery we need.

Reusable pieces:

- ROS 2-based device stack
- camera pipeline
- servo/motor control
- calibration flow
- manual mode
- mode arbitration through `ctrl_pkg`
- start/stop flow through `enable_state`
- existing webserver control pattern
- status and systems services

Important existing control-plane concepts:

- `vehicle_state`: switch active driving mode
- `enable_state`: start or stop the currently selected mode
- `ServoCtrlMsg`: steering + throttle command message

The `Follow the Leader` sample shows that AWS already supports the pattern of adding another mode besides manual/autonomous/calibration.

## Recommended `customrl` MVP

### Goal

Run a custom SB3/PyTorch policy on the physical car while preserving as much of the AWS device stack as possible.

### Core idea

Add a new custom mode named `customrl` with its own inference path.

### Minimal components

1. `customrl_inference_node`
2. `customrl_navigation_node` or direct `ServoCtrlMsg` publisher
3. `customrl` state added to `ctrl_pkg`
4. simple operator interface for:
   - load model
   - arm/select mode
   - start inference
   - stop inference
   - panic stop

### What `customrl_inference_node` should do

- load a model from a file path
- preprocess incoming observations exactly the way the policy expects
- run inference
- convert policy output into steering and throttle values
- publish either:
  - high-level outputs for a navigation node, or
  - direct `ServoCtrlMsg` commands

### What `ctrl_pkg` should do

- treat `customrl` as another mutually exclusive vehicle mode
- allow selecting `customrl` through `vehicle_state`
- allow starting/stopping `customrl` through `enable_state`
- route `customrl` servo messages into the same safe actuation path used by other modes

## TUI and operator control

A simple TUI is feasible and probably the right first operator interface.

The TUI only needs to control the existing ROS services plus the custom model node.

Suggested MVP TUI actions:

- `load model`
- `select customrl mode`
- `start`
- `stop`
- `panic`
- `show status`

Suggested visible status fields:

- current vehicle mode
- whether `customrl` is armed
- selected model path
- inference running or stopped
- latest command timestamp
- watchdog state

## Panic stop requirements

The panic-stop path must **not** depend on the TUI process staying healthy.

The stop path should exist at the ROS/service layer.

Minimum panic behavior:

1. call `enable_state(false)`
2. publish zero-throttle `ServoCtrlMsg`
3. optionally switch back to `manual` mode
4. optionally disable servo GPIO if a harder stop is needed

Additional safety that should exist from day one:

- watchdog that stops the car if inference commands stop arriving
- stop on inference exception
- stop on model-load failure
- stop on stale sensor input

## Why this is better than forcing stock AWS autonomous mode

### Better path

- medium-sized integration project
- clear architecture boundary
- preserves most AWS work
- easier to debug
- easier to turn off
- lower risk of breaking stock features

### Worse path

- reverse-engineer AWS model packaging
- reverse-engineer AWS optimizer and inference assumptions
- reverse-engineer frontend model-management path
- maintain a fake compatibility layer forever

## Realistic effort estimate

Rough effort estimate:

- MVP: a few days to two weeks
- polished version: several weeks
- full AWS-native upload/model-management parity: much larger and higher-risk

This estimate assumes the ROS topics/services on the actual car are visible and behave close to the open-source packages.

## Reasonable deployment strategy

Recommended rollout order:

1. inspect the live ROS topics/services on the car
2. confirm camera input topic and actuation topic/service
3. build `customrl_inference_node`
4. build `customrl` mode in `ctrl_pkg`
5. add ROS-level panic stop
6. add a simple TUI
7. only later consider tighter UI integration

## Open questions before implementation

- what exact DeepRacer device image is on the car
- whether the car is already on the open-source ROS 2 Foxy stack
- what camera topic and message types are present on the real device
- whether direct PyTorch inference is fast enough on-device or whether ONNX/OpenVINO export is needed
- whether we want direct command publication or a separate custom navigation node

## Practical recommendation

The first version should optimize for safety and clarity, not frontend polish.

That means:

- keep the stock AWS autonomous path untouched
- add a parallel `customrl` path
- use a TUI first
- implement panic stop and watchdog before any serious driving

## Sources

- AWS DeepRacer launcher modes of operation:
  - https://github.com/aws-deepracer/aws-deepracer-launcher/blob/main/modes-of-operation.md
- AWS DeepRacer launcher node graph:
  - https://github.com/aws-deepracer/aws-deepracer-launcher/blob/main/deepracer_launcher/launch/deepracer_launcher.py
- AWS DeepRacer systems package:
  - https://github.com/aws-deepracer/aws-deepracer-systems-pkg/blob/main/README.md
- AWS `model_loader_node.py`:
  - https://github.com/aws-deepracer/aws-deepracer-systems-pkg/blob/main/deepracer_systems_pkg/deepracer_systems_pkg/model_loader_module/model_loader_node.py
- AWS DeepRacer Follow the Leader sample:
  - https://github.com/aws-deepracer/aws-deepracer-follow-the-leader-sample-project/blob/main/README.md
- AWS FTL `ctrl_pkg`:
  - https://github.com/aws-deepracer/aws-deepracer-follow-the-leader-sample-project/blob/main/deepracer_follow_the_leader_ws/ctrl_pkg/README.md
- AWS FTL webserver control API:
  - https://github.com/aws-deepracer/aws-deepracer-follow-the-leader-sample-project/blob/main/deepracer_follow_the_leader_ws/webserver_pkg/webserver_pkg/vehicle_control.py
- AWS model import/export docs:
  - https://docs.aws.amazon.com/deepracer/latest/developerguide/import-export-models.html
- DeepRacer on AWS import/export docs:
  - https://docs.aws.amazon.com/solutions/latest/deepracer-on-aws/import-export-models.html
- DeepRacer on AWS vehicle update docs:
  - https://docs.aws.amazon.com/solutions/latest/deepracer-on-aws/update-and-restore-vehicle.html
- AWS blog on open-source device software:
  - https://aws.amazon.com/blogs/machine-learning/aws-deepracer-device-software-now-open-source/
