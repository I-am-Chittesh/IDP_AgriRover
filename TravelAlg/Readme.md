
# AgriTwin rover: boustrophedon coverage path planning, step by step.

Run:
```
    python coverage_demo.py                  # default field, opens 3 figures
    python coverage_demo.py --no-show        # only saves PNGs + waypoints.csv
    python coverage_demo.py --random 5 --seed 7     # random obstacles
    python coverage_demo.py --sweep y        # force sweep direction (x, y or auto)
    python coverage_demo.py --swath 2.0      # change sensor footprint width (m)
```

### Needs only numpy and matplotlib.

## What this script does (same order as the console output):
    STEP 1  Inputs: field rectangle, obstacles, home pose, robot parameters
    STEP 2  Configuration space: shrink the field by the robot radius (headland)
            and inflate every obstacle by the robot radius, so the robot can be
            treated as a single point
    STEP 3  Boustrophedon cell decomposition: sweep a line across the field,
            start a new cell whenever the free interval on the line changes
    STEP 4  Swaths: parallel lines inside each cell, spacing d = footprint * (1 - overlap)
    STEP 5  Mission: order the cells (greedy nearest), snake through each cell,
            connect cells with A* + line-of-sight smoothing, A* back to home
    STEP 6  Verify: rasterize the sensor footprint along the path and measure coverage
    STEP 7  Why it matters: sample a synthetic soil moisture field along the
            path and rebuild a heatmap with IDW (the layer the digital twin shows)

```
In ROS2 this whole pipeline is what opennav_coverage (Fields2Cover) does, and the
final waypoints.csv is what Nav2 FollowWaypoints would consume.
```

