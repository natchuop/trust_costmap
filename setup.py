import os
from glob import glob
from setuptools import setup

package_name = "trust_costmap"

setup(
    name=package_name,
    version="0.0.1",
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
            ],
        ),
        (
            os.path.join("share", package_name, "worlds", "movingai_mapf"),
            glob("worlds/movingai_mapf/*.map"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="vboxuser",
    maintainer_email="vboxuser@example.com",
    description="Trust-weighted costmap experiments using ROS 2 and MovingAI benchmark maps.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "experiment_manager = trust_costmap.experiment_manager_node:main",
        ],
    },
)
