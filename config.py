"""
Configuration Module for UAV-VLA System

This module contains all configuration parameters and templates used throughout
the UAV-VLA (Vision-Language-Action) system.
"""

from typing import Dict, Any

# System Configuration
NUMBER_OF_SAMPLES: int = 30

# Example Data Structures
EXAMPLE_BUILDINGS: Dict[str, Dict[str, Any]] = {
    'building_1': {'type': 'building', 'coordinates': [40.2, 39.5]},
    'building_2': {'type': 'building', 'coordinates': [47.7, 39.0]},
    'building_3': {'type': 'building', 'coordinates': [64.9, 41.2]},
    'building_4': {'type': 'building', 'coordinates': [65.2, 87.9]},
    'building_5': {'type': 'building', 'coordinates': [80.2, 20.7]}
}

example_objects = '''
{
    "village_1": {"type": "village", "coordinates": [1.5, 3.5]},
    "village_2": {"type": "village", "coordinates": [2.5, 6.0]},
    "airfield": {"type": "airfield", "coordinates": [8.0, 6.5]}
}
'''

# Prompt Templates
step_1_template = """
Extract all types of objects the drone needs to find from the following mission description:
"{command}"

Output the result in JSON format with a list of object types.
Example output:
{{
    "object_types": ["village", "airfield", "stadium", "tennis court", "building", "ponds", "crossroad", "roundabout"]
}}
"""

step_3_template = """
Given the mission description: "{command}" and the following identified objects: {objects}, generate a flight plan in pseudo-language.

Available commands:
- arm throttle: arm the copter
- takeoff Z: lift Z meters
- disarm: disarm the copter
- mode rtl: return to home
- mode circle: circle and observe at the current position
- mode guided(X Y Z): fly to the specified location

Example output:
arm throttle
mode guided 43.237763722222226 -85.79224314444444 100
mode guided 43.237765234234234 -85.79224314235235 100
mode circle
mode rtl
disarm
"""

# Default mission command
command = "Create a flight plan for the quadcopter to fly around each of the building at the height 100m return to home and land at the take-off point."

# File paths
BENCHMARK_DIR = "benchmark-UAV-VLPA-nano-30"
IMAGES_DIR = f"{BENCHMARK_DIR}/images"
COORDINATES_FILE = f"{BENCHMARK_DIR}/parsed_coordinates.csv"
MISSION_OUTPUT_DIR = "created_missions"
IDENTIFIED_DATA_DIR = "identified_new_data"