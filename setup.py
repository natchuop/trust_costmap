import os
from glob import glob

from setuptools import setup

package_name = "trust_costmap"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            [
                "package.xml",
                "scenario.yaml",
                "experiment.launch.py",
                "README.md",
                "INSTALL_VM.md",
                "VERIFICATION_REPORT.md",
            ],
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*"),
        ),
        (
            os.path.join("share", package_name, "scripts"),
            glob("scripts/*.py") + glob("scripts/*.sh"),
        ),
        (
            os.path.join("share", package_name, "worlds", "movingai_mapf"),
            glob("worlds/movingai_mapf/*.map"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="natchuop",
    maintainer_email="vboxuser@example.com",
    description=(
        "Trust-weighted costmap experiments with Gazebo multi-robot LiDAR "
        "mapping and RViz visualization."
    ),
    license="MIT",
    entry_points={
        "console_scripts": [
            "experiment_manager = trust_costmap.experiment_manager_node:main",
            "lidar_mapper = trust_costmap.lidar_mapping_node:main",
        ],
    },
)
