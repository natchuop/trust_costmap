# trust_costmap

ROS 2 and Gazebo experiment scaffold for trust-aware occupancy-grid research.
The launch file converts a selected MovingAI map into a Gazebo world, places
TurtleBot3 Burger robots on valid free cells, and scatters visual action goals
throughout the largest connected free-space region.

## Build

```bash
cd <your_ros2_workspace>
colcon build --packages-select trust_costmap --symlink-install
source install/setup.bash
```

## Run from Bash / Linux

```bash
ros2 launch trust_costmap experiment.launch.py \
  map_name:=room-32-32-4 \
  scenario_file:=scenario.yaml \
  planner:=astar \
  allow_diagonal:=false \
  bridge_robot_topics:=true \
  action_goal_count:=8 \
  action_goal_seed:=21 \
  --debug
```

## Run from PowerShell

```powershell
ros2 launch trust_costmap experiment.launch.py `
  map_name:=room-32-32-4 `
  scenario_file:=scenario.yaml `
  planner:=astar `
  allow_diagonal:=false `
  bridge_robot_topics:=true `
  action_goal_count:=8 `
  action_goal_seed:=21 `
  --debug
```

`action_goal_count` controls the exact number of orange action goals generated
in Gazebo and RViz. For example, `action_goal_count:=8` creates exactly eight.

`action_goal_seed` controls deterministic placement of both action goals and
robots. Reusing seed `21` repeats the same layout for the same map and scenario.
Changing the seed changes the layout. A negative seed creates a new random seed
for that launch.

The resolved scenario and generated SDF world are written under
`~/.ros/trust_costmap/`.

## Placement behavior

- Robots and action goals are placed only on free MovingAI cells (`.`, `G`, `S`).
- Robot starts and action goals never share the same cell.
- The largest connected free-space component avoids isolated regions.
- Action goals use seeded farthest-point sampling to spread them across the map.
- Gazebo action goals are small static orange visuals with no collision geometry.
