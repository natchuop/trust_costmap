# trust_costmap

Gazebo/ROS 2 experiments for comparing trust-aware occupancy costmaps. This
version adds a visualization-only multi-robot LiDAR mapping system while leaving
the existing navigation and `/planning_costmap` pipeline unchanged.

## Added LiDAR/RViz behavior

For every enabled robot in `scenario.yaml`, the launch file starts one mapper
that consumes:

- `/<robot_id>/scan`
- `/<robot_id>/odom` as a fallback
- `/<robot_id>/map_pose` from the existing experiment manager

Each mapper publishes:

- `/<robot_id>/current_observation_map`: cells observed by the newest scan
- `/<robot_id>/local_map`: persistent observations made by that robot
- `/<robot_id>/shared_map`: observations received from the other robots
- `/<robot_id>/combined_map`: local plus shared occupancy probabilities
- `/<robot_id>/scan_rviz`: live LiDAR with a namespaced TF frame
- heatmap/history/current-observation marker layers
- a labeled TurtleBot footprint marker

Every robot publishes its local free and occupied observations and subscribes
to every other enabled robot's local map. Combined maps are not re-shared, which
prevents recursive double fusion.

To reduce VM load, only the robot selected by `rviz_robot_id` generates live
scan, TF, heatmap, history, and footprint marker outputs. All robots still build
and exchange occupancy maps.

The single RViz window defaults to the first enabled robot whose ID or role
contains `benign`. Override it with `rviz_robot_id:=<robot_id>`.

## VM setup

See [INSTALL_VM.md](INSTALL_VM.md), or run:

```bash
cd trust_costmap
bash scripts/setup_vm.sh "$HOME/ros_ws"
source "$HOME/ros_ws/install/setup.bash"
```

The setup script detects the sourced ROS distribution and installs the Gazebo,
TurtleBot3, bridge, RViz, colcon, and Python dependencies before building the
package.

## Main run command

```bash
ros2 launch trust_costmap experiment.launch.py \
  costmap_method:=hard_threshold \
  map_name:=room-32-32-4 \
  scenario_file:=scenario.yaml \
  planner:=astar \
  allow_diagonal:=false \
  bridge_robot_topics:=true \
  action_goal_count:=8 \
  action_goal_seed:=21 \
  run_label:=hard_threshold_test \
  --debug
```

LiDAR mapping is enabled by default. `enable_rviz:=auto` opens exactly one RViz
window in GUI runs and disables it automatically for `headless:=true` runs.
The default RViz robot is benign.

Useful overrides:

```bash
# Display another enabled robot.
rviz_robot_id:=benign_2

# Reduce mapper CPU use by processing every second LiDAR beam.
mapping_scan_stride:=2

# Disable only RViz, while all robots continue mapping and sharing.
enable_rviz:=false

# Disable the complete visualization-only mapping subsystem.
enable_lidar_mapping:=false enable_rviz:=false

# Restore full Gazebo materials and shadows.
gazebo_low_graphics:=false
```

## Performance defaults

`gazebo_low_graphics:=true` is enabled by default. The generated world disables
shadows and replaces TurtleBot render meshes / PBR textures with one simple box visual. Robot
models, collision geometry, LiDAR sensors, action goals, and navigation route
visuals remain present.

The LiDAR subsystem publishes persistent maps at 2 Hz. Live scan processing
still follows the Gazebo sensor rate. Increase `mapping_scan_stride` if the VM
is CPU constrained.

## Verification

Run static checks before building:

```bash
bash scripts/verify_project.sh --static-only
```

Run static checks and a package build from a workspace:

```bash
bash scripts/verify_project.sh "$HOME/ros_ws"
```

After launching, useful checks include:

```bash
ros2 topic hz /benign_1/scan
ros2 topic hz /benign_1/local_map
ros2 topic echo /benign_1/map_pose --once
ros2 topic list | grep -E 'current_observation_map|local_map|shared_map|combined_map'
```

## Navigation isolation

The new mapping nodes never publish to:

- `/planning_costmap`
- `/base_map`
- `/map_claims`
- `/trust_updates`
- `/<robot_id>/cmd_vel`

The original experiment manager continues to own planning, route execution,
trust handling, Gazebo route visualization, and experiment logging.
