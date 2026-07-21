# VM installation and test guide

These instructions assume Ubuntu with ROS 2 and Gazebo Sim. The project is
compatible with the existing layout at:

```text
C:\Users\natch\Desktop\School\nsf_reu_shared_vm
```

Inside the Linux VM, place or extract the package into a ROS workspace such as
`~/ros_ws/src/trust_costmap`.

## Automated setup

```bash
cd /path/to/trust_costmap
bash scripts/setup_vm.sh "$HOME/ros_ws"
source "$HOME/ros_ws/install/setup.bash"
```

The script:

1. Detects a sourced ROS distribution, or sources Jazzy/Humble if installed.
2. Installs bootstrap tools and lets `rosdep` resolve ROS/Gazebo packages for the active distribution.
3. initializes `rosdep` when necessary.
4. links the extracted project into the selected workspace.
5. installs missing ROS dependencies with `rosdep`.
6. runs static verification and `colcon build --symlink-install`.
7. verifies that both `experiment_manager` and `lidar_mapper` executables exist.

## Manual setup

```bash
source /opt/ros/jazzy/setup.bash  # replace jazzy when needed
mkdir -p ~/ros_ws/src
cp -a /path/to/trust_costmap ~/ros_ws/src/trust_costmap
cd ~/ros_ws
rosdep install --from-paths src/trust_costmap --ignore-src --rosdistro "$ROS_DISTRO" -r -y
colcon build --symlink-install --packages-select trust_costmap
source install/setup.bash
```

If TurtleBot3 or Gazebo dependencies are missing:

```bash
sudo apt update
sudo apt install -y \
  ros-$ROS_DISTRO-ros-gz-sim \
  ros-$ROS_DISTRO-ros-gz-bridge \
  ros-$ROS_DISTRO-turtlebot3-gazebo \
  ros-$ROS_DISTRO-turtlebot3-description \
  ros-$ROS_DISTRO-rviz2 \
  ros-$ROS_DISTRO-tf2-ros \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-yaml \
  python3-pytest
```

## First test

```bash
source ~/ros_ws/install/setup.bash
bash ~/ros_ws/src/trust_costmap/scripts/run_rviz_demo.sh
```

Expected behavior:

- Gazebo opens with all enabled scenario robots and existing navigation routes.
- One RViz window opens for `benign_1` by default.
- LiDAR points appear around the selected robot.
- current observations appear as a bright overlay.
- previous local observations remain visible.
- observations from other robots appear in the shared layer.
- the combined occupancy map uses probability-dependent heatmap colors.
- a labeled TurtleBot footprint follows the selected robot.

## Selecting the RViz robot

```bash
ros2 launch trust_costmap experiment.launch.py \
  rviz_robot_id:=benign_2 \
  run_label:=rviz_benign_2
```

`rviz_robot_id:=auto` selects the first enabled benign robot. An unknown or
disabled ID produces a launch error listing valid IDs.

## Slow VM settings

The default already uses low-graphics Gazebo rendering with one primitive robot visual. For additional mapper
CPU reduction:

```bash
ros2 launch trust_costmap experiment.launch.py \
  mapping_scan_stride:=2 \
  mapping_publish_rate_hz:=1.0 \
  rviz_robot_id:=auto
```

For automated experiments without GUI rendering:

```bash
ros2 launch trust_costmap experiment.launch.py \
  headless:=true \
  enable_rviz:=auto
```

`enable_rviz:=auto` becomes false in headless mode. LiDAR mapping remains active
unless `enable_lidar_mapping:=false` is also supplied.

## Troubleshooting

### RViz reports no transform

Check the manager pose and mapper TF:

```bash
ros2 topic echo /benign_1/map_pose --once
ros2 run tf2_ros tf2_echo map benign_1/base_scan
```

### No LiDAR data

```bash
ros2 topic hz /benign_1/scan
ros2 topic info /benign_1/scan -v
```

Confirm `bridge_robot_topics:=true` and that the robot ID is enabled in the
selected scenario.

### Maps exist but shared map stays empty

Check that all mapper nodes are running and local maps are publishing:

```bash
ros2 node list | grep lidar_mapper
ros2 topic hz /benign_2/local_map
ros2 topic info /benign_1/shared_map -v
```

### Gazebo graphics driver problems

Try software rendering for a diagnostic run:

```bash
export LIBGL_ALWAYS_SOFTWARE=1
```

This is slower than GPU rendering but can identify VM OpenGL problems.
