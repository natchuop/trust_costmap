# Verification report

This package was checked before archival on July 21, 2026.

## Completed checks

- Python syntax compilation for the launch file, package modules, scripts, and tests.
- XML parsing for `package.xml`.
- YAML parsing for `scenario.yaml` and the RViz template.
- Rendering the RViz template for the default robot and confirming no unresolved placeholders.
- Mapping utility unit tests covering coordinate conversion, odometry conversion, ray tracing, probability conversion, and unknown-cell-preserving fusion.
- Mocked launch integration checks covering:
  - automatic selection of `benign_1`;
  - invalid robot-ID rejection;
  - one mapper per enabled scenario robot;
  - RViz marker output enabled for exactly one selected robot;
  - preservation of Gazebo sensors, collisions, and plugins in low-graphics mode;
  - replacement of render-only robot meshes with one primitive visual.
- Shell syntax checks for all setup and run scripts.
- Python source-distribution and wheel content checks.
- Relative Markdown link validation.

## Navigation isolation checked

The new LiDAR mapper publishes only its own mapping, TF, scan-republish, and visualization topics. It does not publish `/planning_costmap`, `/base_map`, `/map_claims`, `/trust_updates`, or any `cmd_vel` topic. The existing experiment manager remains responsible for planning and movement.

## Runtime limitation of this verification environment

The archive was produced in a container without ROS 2, Gazebo Sim, or RViz installed, so a full graphical simulation could not be executed here. The included `scripts/setup_vm.sh`, `scripts/verify_project.sh`, and `scripts/run_rviz_demo.sh` perform the ROS dependency installation, colcon build, executable check, launch-argument check, and first VM test.
