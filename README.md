# trust_costmap

ROS 2 and Gazebo experiment scaffold for trust-aware occupancy-grid research.
The launch file converts a selected MovingAI map into a Gazebo world, places
TurtleBot3 Burger robots on valid free cells, and scatters visual action-goal
checkpoints throughout the largest connected free-space region.

## Build

```bash
cd <your_ros2_workspace>
colcon build --packages-select trust_costmap --symlink-install
source install/setup.bash
```

## Run

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

`action_goal_seed` controls the complete generated layout. Use the same seed to
repeat a run, or use a negative value to generate a new random seed each time.
The resolved scenario and generated SDF world are written under
`~/.ros/trust_costmap/`.

## Placement behavior

- Robots and checkpoints are placed only on free MovingAI cells (`.`, `G`, `S`).
- All generated cells are unique.
- The largest connected free-space component is used to avoid isolated regions.
- Checkpoints use seeded farthest-point sampling so they are spread across the map.
- Gazebo checkpoints are small static orange visual boxes with no collision geometry.
