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
Generate a flight plan for: "{command}"

Targets: {objects}

CRITICAL REQUIREMENT: Use NEAREST NEIGHBOR algorithm to minimize total travel distance.

OPTIMIZATION ALGORITHM:
1. Calculate distances between all points using coordinates
2. Start from takeoff point
3. Always visit the CLOSEST unvisited target next
4. Continue until all targets visited
5. Return to takeoff point

DISTANCE CALCULATION:
- Use Euclidean distance: sqrt((lat2-lat1)² + (lon2-lon1)²)
- Always choose the target with minimum distance from current position
- Avoid any backtracking or unnecessary detours

Available commands:
- arm throttle: arm the copter
- takeoff Z: lift Z meters
- disarm: disarm the copter
- mode rtl: return to home
- mode circle: circle and observe at the current position
- mode guided(X Y Z): fly to the specified location

OUTPUT FORMAT:
arm throttle
takeoff 100
mode guided [lat] [lon] 100
mode circle
[repeat for each target in optimized order]
mode rtl
disarm

IMPORTANT: Prioritize distance minimization. Visit targets in order of increasing distance from the previous position.
"""

# Default mission command
command = "Create a flight plan for the quadcopter to fly around each of the building at the height 100m return to home and land at the take-off point."

# File paths
BENCHMARK_DIR = "benchmark-UAV-VLPA-nano-30"
IMAGES_DIR = f"{BENCHMARK_DIR}/images"
COORDINATES_FILE = f"{BENCHMARK_DIR}/parsed_coordinates.csv"
MISSION_OUTPUT_DIR = "created_missions"
IDENTIFIED_DATA_DIR = "identified_new_data"