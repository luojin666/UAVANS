"""
优化后的提示词模板，用于生成最短路径的无人机任务计划
"""

# 优化的提示词模板
optimized_step_3_template = """
Given the mission description: "{command}" and the following identified objects: {objects}, generate an OPTIMIZED flight plan that minimizes total travel distance.

CRITICAL OPTIMIZATION REQUIREMENTS:
1. Implement NEAREST NEIGHBOR algorithm for shortest path
2. Calculate distances between ALL points before planning
3. Always visit the CLOSEST unvisited target next
4. Minimize total travel distance
5. Start and end at the same takeoff point
6. Avoid any backtracking or unnecessary detours

DETAILED OPTIMIZATION ALGORITHM:
Step 1: Distance Matrix Calculation
- Calculate Euclidean distance between every pair of points
- Formula: distance = sqrt((lat2-lat1)² + (lon2-lon1)²)
- Include takeoff point in calculations

Step 2: Path Planning (Nearest Neighbor)
- Start from takeoff point
- Find the nearest unvisited target
- Fly to that target
- Mark it as visited
- Repeat until all targets visited
- Return to takeoff point

Step 3: Route Optimization
- Ensure no target is visited twice
- Minimize total distance traveled
- Use direct paths between points

AVAILABLE COMMANDS:
- arm throttle: arm the copter
- takeoff Z: lift Z meters
- disarm: disarm the copter
- mode rtl: return to home
- mode circle: circle and observe at the current position
- mode guided(X Y Z): fly to the specified location

OUTPUT FORMAT:
1. Start with arm throttle
2. Takeoff to specified height
3. Visit targets in optimized order (nearest first)
4. Circle at each target for observation
5. Return to home
6. Disarm

Example optimized output:
arm throttle
takeoff 100
mode guided 43.237763722222226 -85.79224314444444 100
mode circle
mode guided 43.237765234234234 -85.79224314235235 100
mode circle
mode guided 43.237763722222226 -85.79224314444444 100
mode circle
mode rtl
disarm

IMPORTANT: Always prioritize distance minimization over any other consideration.
"""

# 更高级的提示词，包含具体的距离计算示例
advanced_step_3_template = """
Given the mission description: "{command}" and the following identified objects: {objects}, generate an OPTIMIZED flight plan using the NEAREST NEIGHBOR algorithm.

OPTIMIZATION STRATEGY:
1. Calculate distances between all points using coordinates
2. Start from takeoff point
3. Always choose the nearest unvisited target
4. Continue until all targets visited
5. Return to takeoff point

DISTANCE CALCULATION METHOD:
For each pair of points (lat1,lon1) and (lat2,lon2):
distance = sqrt((lat2-lat1)² + (lon2-lon1)²)

ALGORITHM STEPS:
1. List all targets with their coordinates
2. Start at takeoff point (use first target as reference)
3. Find the target with minimum distance from current position
4. Add that target to the route
5. Mark it as visited
6. Set current position to that target
7. Repeat steps 3-6 until all targets visited
8. Add return to takeoff point

Available commands:
- arm throttle: arm the copter
- takeoff Z: lift Z meters
- disarm: disarm the copter
- mode rtl: return to home
- mode circle: circle and observe at the current position
- mode guided(X Y Z): fly to the specified location

Generate the flight plan with targets visited in order of increasing distance from the previous position.
"""

# 简化版本，专注于核心优化
simple_optimized_template = """
Generate a flight plan for: "{command}"

Targets: {objects}

REQUIREMENT: Use NEAREST NEIGHBOR algorithm to minimize total travel distance.

Algorithm:
1. Start from takeoff point
2. Visit the closest unvisited target
3. Repeat until all targets visited
4. Return to takeoff point

Available commands:
- arm throttle: arm the copter
- takeoff Z: lift Z meters
- disarm: disarm the copter
- mode rtl: return to home
- mode circle: circle and observe at the current position
- mode guided(X Y Z): fly to the specified location

Generate the optimized flight plan.
""" 