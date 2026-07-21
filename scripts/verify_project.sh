#!/usr/bin/env bash
set -eo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
WORKSPACE="${1:-$HOME/ros_ws}"
if [[ "${MODE}" == "--static-only" ]]; then
  WORKSPACE=""
fi

cd "${PACKAGE_DIR}"

required_files=(
  package.xml
  setup.py
  experiment.launch.py
  scenario.yaml
  README.md
  INSTALL_VM.md
  VERIFICATION_REPORT.md
  config/lidar_mapping.rviz.in
  trust_costmap/experiment_manager_node.py
  trust_costmap/lidar_mapping_node.py
  trust_costmap/mapping_utils.py
  scripts/check_launch_helpers.py
  scripts/setup_vm.sh
  scripts/run_rviz_demo.sh
)
for path in "${required_files[@]}"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done

bash -n scripts/setup_vm.sh scripts/run_rviz_demo.sh

export PYTHONDONTWRITEBYTECODE=1
python3 - <<'PY'
from pathlib import Path
import ast

for path in (
    "experiment.launch.py",
    "malicious_object_node.py",
    "visualize_map.py",
    *Path("trust_costmap").glob("*.py"),
    *Path("scripts").glob("*.py"),
    *Path("test").glob("*.py"),
):
    ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
print("Python syntax checks passed")
PY

python3 - <<'PY'
from pathlib import Path
import ast
import re
import xml.etree.ElementTree as ET
import yaml

ET.parse("package.xml")
yaml.safe_load(Path("scenario.yaml").read_text(encoding="utf-8"))
template = Path("config/lidar_mapping.rviz.in").read_text(encoding="utf-8")
assert "__ROBOT_ID__" in template
assert "__PEER_SCAN_DISPLAYS__" in template
for required in (
    "/rviz_base_free",
    "/rviz_base_occupied",
    "/__ROBOT_ID__/lidar_free_cells",
    "/__ROBOT_ID__/lidar_occupied_cells",
    "/__ROBOT_ID__/scan_rviz",
    "/agent_routes/__ROBOT_ID__",
):
    assert required in template, required
# Placeholder line is not valid YAML until launch expands it.
sample = (
    template.replace("__ROBOT_ID__", "benign_1").replace(
        "__PEER_SCAN_DISPLAYS__", ""
    )
)
yaml.safe_load(sample)
assert "rviz_default_plugins/Map" not in sample
assert "rviz_default_plugins/GridCells" in sample

for document in (
    Path("README.md"),
    Path("INSTALL_VM.md"),
    Path("VERIFICATION_REPORT.md"),
):
    text = document.read_text(encoding="utf-8")
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        resolved = (document.parent / target.split("#", 1)[0]).resolve()
        assert resolved.exists(), f"Broken relative link in {document}: {target}"

mapper_path = Path("trust_costmap/lidar_mapping_node.py")
mapper_tree = ast.parse(mapper_path.read_text(encoding="utf-8"))
forbidden_publish_topics = {
    "/planning_costmap",
    "/base_map",
    "/map_claims",
    "/trust_updates",
}
for node in ast.walk(mapper_tree):
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        continue
    if node.func.attr != "create_publisher" or len(node.args) < 2:
        continue
    topic_argument = node.args[1]
    if isinstance(topic_argument, ast.Constant) and isinstance(topic_argument.value, str):
        assert topic_argument.value not in forbidden_publish_topics
        assert not topic_argument.value.endswith("/cmd_vel")

setup_text = Path("setup.py").read_text(encoding="utf-8")
assert 'lidar_mapper = trust_costmap.lidar_mapping_node:main' in setup_text
assert '"VERIFICATION_REPORT.md"' in setup_text
launch_text = Path("experiment.launch.py").read_text(encoding="utf-8")
assert 'default_value="auto"' in launch_text
assert 'gazebo_low_graphics' in launch_text
assert 'enable_rviz_outputs' in launch_text
manager_text = Path("trust_costmap/experiment_manager_node.py").read_text(encoding="utf-8")
assert 'f"/{robot.robot_id}/map_pose"' in manager_text

print("XML/YAML/link/navigation-isolation checks passed")
PY

PYTHONPATH=. python3 -m pytest -q test/test_mapping_utils.py
python3 scripts/check_launch_helpers.py

echo "Static project verification passed."

if [[ -n "${WORKSPACE}" ]]; then
  if [[ -z "${ROS_DISTRO:-}" ]]; then
    echo "ROS_DISTRO is not set. Source /opt/ros/<distro>/setup.bash first." >&2
    exit 1
  fi
  cd "${WORKSPACE}"
  colcon build --symlink-install --packages-select trust_costmap
  # shellcheck disable=SC1090
  source "${WORKSPACE}/install/setup.bash"
  ros2 pkg executables trust_costmap | grep -q 'trust_costmap experiment_manager'
  ros2 pkg executables trust_costmap | grep -q 'trust_costmap lidar_mapper'
  ros2 launch trust_costmap experiment.launch.py --show-args >/dev/null
  echo "ROS package build and launch-argument verification passed."
fi
