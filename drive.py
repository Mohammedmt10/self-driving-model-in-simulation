import os
import sys
import math
import time
import queue
import random
import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import carla


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_FILE = os.path.join(
    PROJECT_DIR,
    "model_v15.pth"
)

DRIVELM_ROOT = os.path.expanduser(
    "~/DriveLM-CARLA"
)

PDM_ROOT = os.path.join(
    DRIVELM_ROOT,
    "pdm_lite"
)

PDM_TEAM_CODE = os.path.join(
    PDM_ROOT,
    "team_code"
)

CARLA_ROOT = os.path.expanduser(
    "~/carla-simulator"
)

CARLA_PYTHON_API = os.path.join(
    CARLA_ROOT,
    "PythonAPI",
    "carla"
)


for path in (
    PDM_TEAM_CODE,
    CARLA_PYTHON_API
):
    if (
        os.path.isdir(path)
        and path not in sys.path
    ):
        sys.path.insert(
            0,
            path
        )


# ============================================================
# IMPORTS
# ============================================================

try:

    import transfuser_utils as t_u

    from agents.navigation.global_route_planner import (
        GlobalRoutePlanner
    )

    from agents.navigation.local_planner import (
        RoadOption
    )

except Exception as exc:

    raise RuntimeError(
        "\nCould not import CARLA/PDM-Lite modules.\n"
        f"PDM path: {PDM_TEAM_CODE}\n"
        f"Original error: {exc}"
    )


from model import AutonomousDriver


# ============================================================
# CONFIG
# ============================================================

CARLA_HOST = "localhost"
CARLA_PORT = 2000

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_WIDTH = 480
MODEL_HEIGHT = 240
MODEL_TELEMETRY = 16

# Fraction of the top of each camera frame to drop before feeding
# the model, so inference matches dataset.py.  The dataset crops the
# top 30% (sky / buildings / trees) during training:
#     image = image[int(0.3 * image.shape[0]):, :]
IMAGE_CROP_RATIO = 0.30


# ============================================================
# CAMERA
#
# DO NOT CHANGE
# ============================================================

CAMERA_X = -1.5
CAMERA_Y = 0.0
CAMERA_Z = 2.0

CAMERA_ROLL = 0.0
CAMERA_PITCH = 0.0
CAMERA_YAW = 0.0

CAMERA_WIDTH = 1024
CAMERA_HEIGHT = 512
CAMERA_FOV = 110.0


# ============================================================
# SIMULATION
# ============================================================

FIXED_DELTA_SECONDS = 0.05
IMU_SENSOR_TICK = 0.05


# ============================================================
# VEHICLE
# ============================================================

VEHICLE_FILTER = "vehicle.tesla.model3"


# ============================================================
# TRAFFIC
#
# How much NPC traffic/objects to place in the world when driving.
# ============================================================

TRAFFIC_VEHICLES = 40      # autopilot NPC cars placed on the map
TRAFFIC_STATIC_PROPS = 10  # static scenery props beside the road
TRAFFIC_PORT = 8000        # TrafficManager port (server-side)


# ============================================================
# CONTROL
# ============================================================

BRAKE_THRESHOLD = 0.50
MAX_THROTTLE = 0.60

# --- Steering Stability & Smoothness ---
# Simplified single-stage EMA smoothing.  The previous double-EMA + rate
# limiter + speed-adaptive sensitivity chain created so much lag that the
# model's corrections never reached the wheels in time, causing the car
# to spin in place or drive straight off-route.
STEER_SMOOTHING = 0.40          # EMA smoothing factor (lower = more responsive)
MAX_STEER_STEP = 0.15           # Max change per frame (rate limit)
STEER_DEAD_ZONE = 0.01          # Dead zone around center to prevent jitter


# ============================================================
# DISPLAY
# ============================================================

SHOW_CAMERA = True
PRINT_TELEMETRY = True


# ============================================================
# TOWNS
# ============================================================
TOWNS = [
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10HD",
]

# Towns known to be heavy on VRAM or prone to segfaults on <8GB GPUs.
# Used only for warning messages; the actual probe happens at runtime.
HEAVY_TOWNS = {
    "Town03",
    "Town04",
    "Town10HD",
}

# Towns confirmed working for this install (updated by --test-towns).
CONFIRMED_FILE = os.path.join(
    PROJECT_DIR,
    ".confirmed_towns"
)


# ============================================================
# ROUTE SELECTION
# ============================================================

MIN_ROUTE_DISTANCE = 180.0
MAX_ROUTE_DISTANCE = 1500.0
DESTINATION_ATTEMPTS = 150
REQUIRE_TURN = True

# Spawn-validity tolerances.
#
# Root cause (seen in the wild): if the random spawn point is far from the
# drivable road graph, GlobalRoutePlanner snaps the route start a long way
# from the vehicle (e.g. ~294 m). ensure_route_starts_at_vehicle() then
# prepends the car's transform, producing a huge first segment that points
# ~169deg off the car's heading, so the angle input fed to the model says
# "the route is behind me" and the car just wheels sideways and gets stuck.
#
#   SPAWN_ROAD_TOLERANCE : max distance (m) from a candidate spawn point to
#       the nearest drivable road centerline. Spawn points farther away are
#       skipped so the ego vehicle always starts near the drivable network.
#   ROUTE_START_TOLERANCE: max distance (m) allowed between the vehicle and
#       route[0] (the planner's snapped start). If the planner had to snap
#       farther than this, the route is rejected (backstop for a non-drivable
#       spawn). Must exceed the planner node_density (2.0) plus spawn offset.
SPAWN_ROAD_TOLERANCE = 3.0
ROUTE_START_TOLERANCE = 5.0

# Max heading error (rad) allowed between the vehicle and the start of a
# chosen route. If the first real route point lies more than this off the
# vehicle's heading, the route is rejected outright — otherwise the model
# is told "the route is behind me" and the car wheels hard to one side and
# spins in place.
FORWARD_ANGLE_TOLERANCE = 1.2   # ~69 deg

# Distance (m) from the destination at which the driver brakes and stops.
# Without this the car never terminates: it keeps driving and, once past the
# end of the route, chases the final route point (now behind it) in circles.
DESTINATION_REACHED_DISTANCE = 10.0


# ============================================================
# COMMAND NAMES
# ============================================================

COMMAND_NAMES = {
    1: "LEFT",
    2: "RIGHT",
    3: "STRAIGHT",
    4: "LANEFOLLOW",
    5: "CHANGE_LANE_LEaT",
    6: "CHANGE_LANE_RIGHT",
}



# ============================================================
# IMAGE
#
# Single source of truth for what the driver feeds the model.
# IMAGE_SPEC documents the pipeline; IMAGE_TRANSFORM is built
# FROM it, so the code and its description cannot drift apart.
#
# Pipeline (must match dataset.py CarlaDataset image load path):
#
#   [CARLA RGB uint8, 512 x 1024]
#       -> drop top 30%            (sky / buildings / trees)
#       -> grayscale (luminance)
#       -> resize to 480 x 240     (H, W)
#       -> ToTensor -> float32 tensor (1, 240, 480), values in [0, 1]
#       -> NO mean/std normalization (same as dataset.py)
# ============================================================

IMAGE_SPEC = {
    # --- input ---
    "input": {
        "channels": "RGB",
        "dtype": "uint8",
        "range": [0, 255],
        "height": CAMERA_HEIGHT,
        "width": CAMERA_WIDTH,
    },
    # --- crop ---
    "crop": {
        "ratio": IMAGE_CROP_RATIO,
        "px": int(
            IMAGE_CROP_RATIO * CAMERA_HEIGHT
        ),
        "note": (
            "top of the frame is removed before resize, "
            "exactly like dataset.py"
        ),
    },
    # --- grayscale ---
    "grayscale": True,
    "grayscale_formula": (
        "luminance: 0.299 R + 0.587 G + 0.114 B"
    ),
    # --- resize ---
    "resize": (MODEL_HEIGHT, MODEL_WIDTH),   # (H, W)
    "resize_interpolation": (
        "BILINEAR (torchvision Resize default)"
    ),
    # --- output tensor ---
    "output": {
        "channels": 1,
        "shape": (1, MODEL_HEIGHT, MODEL_WIDTH),
        "dtype": "float32",
        "range": [0.0, 1.0],
        "normalization": "none (raw [0,1])",
    },
    # --- reference statistics ---
    "note": (
        "IMG_MEAN / IMG_STD computed over the training set "
        "(0.4020 / 0.1886) are NOT applied here.  Applying them "
        "without matching dataset.py would desync inference from "
        "training."
    ),
}


IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize(
        IMAGE_SPEC["resize"]
    ),
    transforms.ToTensor(),
])


def report_image_spec(title="DRIVER INFERENCE IMAGE SPEC"):

    """
    Print the exact image format the driver produces, so it can
    be compared against the dataset load path without reading
    either implementation.
    """

    print()
    print(title)
    print("-" * 70)

    for key, value in IMAGE_SPEC.items():

        if isinstance(value, dict):

            print(f"{key}:")

            for sub_key, sub_value in value.items():

                print(
                    f"    {sub_key:<20}: {sub_value}"
                )

        else:

            print(
                f"{key:<20}: {value}"
            )

    print("-" * 70)


# ============================================================
# SENSOR QUEUES
# ============================================================

camera_queue = queue.Queue()
imu_queue = queue.Queue()


# ============================================================
# STEERING STATE
# ============================================================

_last_steering = 0.0
_last_steering_smoothed = 0.0


# ============================================================
# GENERAL HELPERS
# ============================================================

def command_name(command):

    try:
        return COMMAND_NAMES.get(
            int(command),
            "UNKNOWN"
        )
    except Exception:
        return "UNKNOWN"


# ============================================================

def safe_float(
    value,
    name,
    fallback=None
):

    if value is None:

        if fallback is not None:
            return float(fallback)

        raise RuntimeError(
            f"{name} is None."
        )

    try:
        result = float(value)
    except (
        TypeError,
        ValueError
    ):

        if fallback is not None:
            return float(fallback)

        raise RuntimeError(
            f"{name} is not numeric: {value!r}"
        )

    if not math.isfinite(result):

        if fallback is not None:
            return float(fallback)

        raise RuntimeError(
            f"{name} is not finite: {result}"
        )

    return result


# ============================================================

def wrap_angle(angle):

    return (
        (
            angle + math.pi
        )
        %
        (
            2.0 * math.pi
        )
    ) - math.pi


# ============================================================

def camera_callback(image):
    camera_queue.put(image)


# ============================================================

def imu_callback(measurement):
    imu_queue.put(measurement)


# ============================================================

def get_sensor_frame_data(
    sensor_queue,
    frame
):

    while True:

        try:
            data = sensor_queue.get(
                timeout=3.0
            )

        except queue.Empty:

            raise RuntimeError(
                f"Timed out waiting for sensor frame {frame}."
            )

        if data.frame < frame:
            continue

        # If a newer frame has already arrived (we fell behind — a slow
        # inference frame, a missed tick, a rendering stall), skip ahead
        # to it rather than crash the whole driver.  Camera and IMU are
        # polled for the same tick, so they stay roughly aligned.
        return data


# ============================================================

def image_to_rgb(image):

    raw = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    raw = raw.reshape(
        image.height,
        image.width,
        4
    )

    # BGRA -> RGB
    return raw[:, :, :3][:, :, ::-1].copy()


# ============================================================

def get_speed_ms(vehicle):

    return safe_float(
        vehicle
        .get_velocity()
        .length(),
        "vehicle speed",
        0.0
    )


# ============================================================

def get_carla_speed_limit_ms(vehicle):

    try:
        value = vehicle.get_speed_limit()
    except Exception:
        value = None

    if value is None:
        return 50.0 / 3.6

    value = safe_float(
        value,
        "CARLA speed limit",
        50.0
    )

    if value <= 0.0:
        return 50.0 / 3.6

    return value / 3.6


# ============================================================

def get_pdm_theta(
    imu_measurement
):

    compass = safe_float(
        imu_measurement.compass,
        "IMU compass"
    )

    theta = safe_float(
        t_u.preprocess_compass(
            compass
        ),
        "processed PDM theta"
    )

    return (
        compass,
        theta
    )


# ============================================================
# WORLD -> EGO
#
# X = forward
# Y = right
# ============================================================

def world_to_ego_xy(
    vehicle,
    world_x,
    world_y
):

    transform = vehicle.get_transform()

    dx = (
        float(world_x)
        -
        float(transform.location.x)
    )

    dy = (
        float(world_y)
        -
        float(transform.location.y)
    )

    yaw = math.radians(
        float(
            transform.rotation.yaw
        )
    )

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    local_x = (
        cos_yaw * dx
        +
        sin_yaw * dy
    )

    local_y = (
        -sin_yaw * dx
        +
        cos_yaw * dy
    )

    return (
        local_x,
        local_y
    )


# ============================================================
# WORLD
# ============================================================

def load_navigation_town(
    client,
    town
):

    if town not in TOWNS:

        raise RuntimeError(
            f"Invalid town {town}. "
            f"Use one of {TOWNS}."
        )

    print()
    print("=" * 82)
    print("LOADING NAVIGATION TOWN")
    print("=" * 82)

    print(
        "Selected town:",
        town
    )

    world = client.load_world(
        town
    )

    client.set_timeout(
        10.0
    )

    expected_map = (
        f"Carla/Maps/{town}"
    )

    start_time = time.time()

    while (
        time.time() - start_time
        <
        30.0
    ):

        current_map = (
            world.get_map().name
        )

        if current_map == expected_map:

            print(
                "Active map:",
                current_map
            )

            return world

        time.sleep(
            0.1
        )

    raise RuntimeError(
        f"Failed to load {town}.\n"
        f"Expected: {expected_map}\n"
        f"Current: {world.get_map().name}"
    )


# ============================================================
# VEHICLE
# ============================================================

def spawn_vehicle(world):

    blueprints = (
        world
        .get_blueprint_library()
        .filter(VEHICLE_FILTER)
    )

    if not blueprints:

        raise RuntimeError(
            f"Vehicle blueprint not found: "
            f"{VEHICLE_FILTER}"
        )

    spawn_points = list(
        world
        .get_map()
        .get_spawn_points()
    )

    if not spawn_points:

        raise RuntimeError(
            "No CARLA spawn points."
        )

    random.shuffle(
        spawn_points
    )

    print(
        f"Available spawn points: {len(spawn_points)} "
        f"(picking a random one each time)."
    )

    carla_map = world.get_map()

    for transform in spawn_points:

        wp = carla_map.get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if wp is None or wp.transform.location.distance(transform.location) > SPAWN_ROAD_TOLERANCE:
            continue

        vehicle = (
            world
            .try_spawn_actor(
                blueprints[0],
                transform
            )
        )

        if vehicle is not None:

            vehicle.set_autopilot(
                False
            )
            
            world.tick()

            print()
            print(
                "Spawned vehicle:"
            )

            print(
                "  Type:",
                vehicle.type_id
            )

            print(
                "  ID:",
                vehicle.id
            )

            print(
                "  Location:",
                vehicle.get_location()
            )

            print(
                "  Rotation:",
                vehicle
                .get_transform()
                .rotation
            )

            return vehicle

    raise RuntimeError(
        "Could not spawn ego vehicle."
    )


# ============================================================
# TRAFFIC
#
# Populate the world with NPC cars and a little static scenery so
# the ego drives in a populated scene.  Purely additive: the ego and
# its route are untouched.  Cars are handed to the TrafficManager
# (which runs server-side), so they keep driving after setup.
# ============================================================

def spawn_traffic(
    world,
    client,
    ego,
    count=TRAFFIC_VEHICLES,
    props=TRAFFIC_STATIC_PROPS,
    tm_port=TRAFFIC_PORT
):

    """Spawn `count` autopilot NPC vehicles + `props` static objects.

    Skips the ego's own blueprint and any spawn point within 40 m of it.
    Returns the number of vehicles spawned.
    """

    bp_lib = world.get_blueprint_library()

    # --- NPC vehicles (autopilot via TrafficManager) ---
    veh_bps = [
        b
        for b in bp_lib.filter("vehicle.*")
        if b.id != VEHICLE_FILTER            # never a mirror of the ego
        and b.has_attribute("number_of_wheels")
        and b.get_attribute("number_of_wheels").as_int() == 4
    ]

    if not veh_bps:
        print(
            "spawn_traffic: no NPC vehicle blueprints "
            "found; skipping."
        )
        return 0

    spawn_pts = list(
        world.get_map().get_spawn_points()
    )
    random.shuffle(spawn_pts)

    ego_loc = ego.get_location() if ego is not None else None

    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(
        world.get_settings().synchronous_mode
    )
    tm.global_percentage_speed_difference(10.0)
    tm.set_global_distance_to_leading_vehicle(2.5)

    spawned = 0
    failures = 0
    for pt in spawn_pts:
        if spawned >= count:
            break
        if (
            ego_loc is not None
            and pt.location.distance(ego_loc) < 40.0
        ):
            continue
        actor = world.try_spawn_actor(
            random.choice(veh_bps), pt
        )
        if actor is None:
            failures += 1
            continue
        actor.set_autopilot(True, tm_port)
        spawned += 1

    print(
        f"Traffic: spawned {spawned} NPC vehicles "
        f"({failures} spawn failures)."
    )

    # --- static scenery props beside the road ---
    prop_bps = bp_lib.filter("static.prop.*")
    if prop_bps and props > 0:
        wps = world.get_map().generate_waypoints(60.0)
        random.shuffle(wps)
        p_spawned = 0
        for wp in wps:
            if p_spawned >= props:
                break
            t = wp.transform
            yaw = math.radians(t.rotation.yaw + 90.0)
            side = random.choice((-5.0, 5.0))
            loc = carla.Location(
                x=t.location.x + side * math.cos(yaw),
                y=t.location.y + side * math.sin(yaw),
                z=t.location.z + 0.2,
            )
            actor = world.try_spawn_actor(
                random.choice(prop_bps),
                carla.Transform(loc, t.rotation),
            )
            if actor is None:
                continue
            p_spawned += 1
        print(
            f"Traffic: placed {p_spawned} static props."
        )

    return spawned


# ============================================================
# ROUTE NORMALIZATION
#
# GlobalRoutePlanner returns:
#
#     (Waypoint, RoadOption)
#
# The rest of this driver uses:
#
#     (Transform, RoadOption)
#
# This fixes the Waypoint.location error.
# ============================================================

def normalize_route(route):

    if route is None:
        return []

    normalized = []

    for index, item in enumerate(route):

        if (
            not isinstance(
                item,
                (tuple, list)
            )
            or
            len(item) != 2
        ):

            raise RuntimeError(
                f"Invalid route item "
                f"at index {index}: {item!r}"
            )

        position, road_option = item

        # Waypoint -> Transform
        if isinstance(
            position,
            carla.Waypoint
        ):

            transform = position.transform

        # Already Transform
        elif isinstance(
            position,
            carla.Transform
        ):

            transform = position

        # Something containing transform
        elif hasattr(
            position,
            "transform"
        ):

            transform = position.transform

        else:

            raise RuntimeError(
                f"Invalid route position "
                f"at index {index}: "
                f"{type(position).__name__}"
            )

        normalized.append(
            (
                transform,
                road_option
            )
        )

    return normalized


# ============================================================
# ROUTE HELPERS
# ============================================================

def route_length(route):

    if (
        route is None
        or
        len(route) < 2
    ):

        return 0.0

    total = 0.0

    for i in range(
        1,
        len(route)
    ):

        p0 = route[
            i - 1
        ][0].location

        p1 = route[
            i
        ][0].location

        total += p0.distance(
            p1
        )

    return total


# ============================================================

def route_has_turn(route):

    return any(
        option in (
            RoadOption.LEFT,
            RoadOption.RIGHT
        )
        for _, option in route
    )


# ============================================================

def route_turn_summary(route):

    turns = []
    last = None

    for _, option in route:

        if option == RoadOption.LEFT:

            name = "LEFT"

        elif option == RoadOption.RIGHT:

            name = "RIGHT"

        else:

            last = None
            continue

        if name != last:
            turns.append(name)

        last = name

    return turns


# ============================================================
# DESTINATION
# ============================================================

def choose_destination(
    world,
    vehicle
):

    spawn_points = list(
        world
        .get_map()
        .get_spawn_points()
    )

    if len(spawn_points) < 2:

        raise RuntimeError(
            "Not enough CARLA spawn points."
        )

    start = (
        vehicle
        .get_transform()
    )

    candidates = list(
        spawn_points
    )

    random.shuffle(
        candidates
    )

    fallback = None

    planner = GlobalRoutePlanner(
        world.get_map(),
        2.0
    )

    print()
    print("=" * 82)
    print(
        "SEARCHING FOR NAVIGATION DESTINATION"
    )
    print("=" * 82)

    for attempt in range(
        DESTINATION_ATTEMPTS
    ):

        destination = random.choice(
            candidates
        )

        straight_distance = (
            start.location.distance(
                destination.location
            )
        )

        if (
            straight_distance
            <
            MIN_ROUTE_DISTANCE
        ):
            continue

        if (
            straight_distance
            >
            MAX_ROUTE_DISTANCE
        ):
            continue

        try:

            raw_route = planner.trace_route(
                start.location,
                destination.location
            )

            # IMPORTANT FIX
            route = normalize_route(
                raw_route
            )

        except Exception:

            continue

        if len(route) < 10:
            continue

        first_wp_transform = route[0][0]
        start_dist = start.location.distance(first_wp_transform.location)
        if start_dist > ROUTE_START_TOLERANCE:
            print(f"DEBUG: rejected due to start_dist ({start_dist:.2f}) > ROUTE_START_TOLERANCE ({ROUTE_START_TOLERANCE})")
            continue

        route_yaw = math.radians(first_wp_transform.rotation.yaw)
        vehicle_yaw = math.radians(start.rotation.yaw)
        diff = abs(wrap_angle(route_yaw - vehicle_yaw))
        if diff > FORWARD_ANGLE_TOLERANCE:
            print(f"DEBUG: rejected due to angle diff ({diff:.2f}) > FORWARD_ANGLE_TOLERANCE ({FORWARD_ANGLE_TOLERANCE})")
            continue

        length = route_length(
            route
        )

        if (
            length
            <
            MIN_ROUTE_DISTANCE
        ):
            continue

        if fallback is None:

            fallback = (
                destination,
                route,
                length
            )

        if (
            REQUIRE_TURN
            and
            not route_has_turn(route)
        ):
            continue

        print(
            f"Destination found on attempt "
            f"{attempt + 1}."
        )

        print(
            f"Straight-line distance: "
            f"{straight_distance:.1f} m"
        )

        print(
            f"Route distance: "
            f"{length:.1f} m"
        )

        print(
            "Route decisions:",
            route_turn_summary(route)
            or
            ["STRAIGHT"]
        )

        return (
            destination,
            route
        )

    if fallback is None:

        raise RuntimeError(
            "Could not find a valid "
            "destination route."
        )

    (
        destination,
        route,
        length
    ) = fallback

    print(
        "No route with LEFT/RIGHT found; "
        "using fallback route."
    )

    print(
        f"Fallback route distance: "
        f"{length:.1f} m"
    )

    print(
        "Route decisions:",
        route_turn_summary(route)
        or
        ["STRAIGHT"]
    )

    return (
        destination,
        route
    )


# ============================================================
# ROUTE INITIALIZATION
# ============================================================

def ensure_route_starts_at_vehicle(vehicle, route):
    """
    Ensure the route starts at the vehicle's current location.
    The GlobalRoutePlanner returns a route from the road network,
    which may not start exactly at the vehicle's position.

    If the first route point is only a short hop away we prepend the
    vehicle transform (a small offset is fine — that segment then points
    roughly forward). If the first route point is far away, the planner
    snapped the start to a distant node; prepending would create a giant
    phantom segment pointing behind the car and make the model chase it.
    Instead we trim the leading points up to the nearest point that lies
    ahead of the vehicle, then prepend a clean short segment.
    """
    if not route or len(route) < 2:
        return route

    vehicle_transform = vehicle.get_transform()
    first_wp_transform = route[0][0]

    # Distance from vehicle to first route point
    dist = vehicle_transform.location.distance(first_wp_transform.location)

    # Small offset: prepend the vehicle position (LANEFOLLOW first segment).
    if dist <= ROUTE_START_TOLERANCE:
        if dist > 2.0:
            print(f"  Route starts {dist:.1f}m from vehicle, prepending vehicle transform")
            return [(vehicle_transform, RoadOption.LANEFOLLOW)] + route
        return route

    # Large offset: trim the leading points up to the nearest forward point.
    print(f"  Route starts {dist:.1f}m from vehicle; trimming leading points")

    best_index = None
    best_forward_x = float("inf")

    for i in range(1, len(route)):
        lx, ly = world_to_ego_xy(
            vehicle,
            route[i][0].location.x,
            route[i][0].location.y
        )
        if lx <= 0.0:
            continue
        if lx < best_forward_x:
            best_forward_x = lx
            best_index = i

    if best_index is not None:
        trimmed = route[best_index:]
        return [(vehicle_transform, RoadOption.LANEFOLLOW)] + trimmed

    return route


# ============================================================
# DIRECT CARLA ROUTE STATE
# ============================================================

class CarlaRouteState:

    def __init__(
        self,
        world,
        vehicle,
        route
    ):

        self.world = world
        self.vehicle = vehicle
        self.route = route

        # route is now guaranteed to be:
        # (Transform, RoadOption)

        self.route_points = np.array(
            [
                [
                    transform.location.x,
                    transform.location.y,
                    transform.location.z
                ]
                for transform, _ in route
            ],
            dtype=np.float64
        )

        self.route_options = [
            option
            for _, option in route
        ]

        self.nearest_index = 0

        self.aim_point = None

        self.command = 4
        self.next_command = 4

        self.speed_limit = (
            50.0 / 3.6
        )

        self.lateral_raw = 0.0

        self.angle_raw = 0.0


    # ========================================================
    # COMMAND
    # ========================================================

    @staticmethod
    def option_to_command(option):

        if option == RoadOption.LEFT:
            return 1

        if option == RoadOption.RIGHT:
            return 2

        if option == RoadOption.STRAIGHT:
            return 3

        if option == RoadOption.LANEFOLLOW:
            return 4

        if option == RoadOption.CHANGELANELEFT:
            return 5

        if option == RoadOption.CHANGELANERIGHT:
            return 6

        return 4


    # ========================================================
    # NEAREST SEGMENT
    # ========================================================

    def find_nearest_segment(self):

        route = self.route_points

        if len(route) < 2:

            return (
                0,
                0.0
            )

        location = (
            self.vehicle
            .get_location()
        )

        ego = np.array(
            [
                location.x,
                location.y
            ],
            dtype=np.float64
        )

        best_distance_sq = float(
            "inf"
        )

        best_index = 0
        best_t = 0.0

        # Scan the ENTIRE route for the true nearest segment.  Searching
        # only a forward window made nearest_index monotonic: after a
        # collision, a lane swap, or a loop (roundabout) the vehicle could
        # never re-find the correct segment and locked onto a far-ahead one,
        # producing a wrong angle and wrong commands.  The prepended
        # vehicle-position segment is short and never wins, so it cannot
        # trap the search.
        for i in range(
            len(route) - 1
        ):

            p1 = route[
                i,
                :2
            ]

            p2 = route[
                i + 1,
                :2
            ]

            segment = (
                p2 - p1
            )

            length_sq = float(
                np.dot(
                    segment,
                    segment
                )
            )

            if length_sq <= 1e-9:

                continue

            t = float(
                np.dot(
                    ego - p1,
                    segment
                )
                /
                length_sq
            )

            t = float(
                np.clip(
                    t,
                    0.0,
                    1.0
                )
            )

            closest = (
                p1
                +
                t * segment
            )

            delta = (
                ego
                -
                closest
            )

            distance_sq = float(
                np.dot(
                    delta,
                    delta
                )
            )

            if distance_sq < best_distance_sq:

                best_distance_sq = (
                    distance_sq
                )

                best_index = i

                best_t = t

        self.nearest_index = (
            best_index
        )

        return (
            best_index,
            best_t
        )


    # ========================================================
    # LOOKAHEAD
    # ========================================================

    def get_lookahead_distance(self):

        speed_kmh = (
            get_speed_ms(
                self.vehicle
            )
            *
            3.6
        )

        lookahead = (
            4.0
            +
            0.20 * speed_kmh
        )

        return float(
            np.clip(
                lookahead,
                4.0,
                12.0
            )
        )

    # --------------------------------------------------------
    # TARGET POINT (for angle) — speed-scaled lookahead matching
    # the PDM-Lite data collector (`autopilot.py` `_get_steer`).
    #
    # The collector computes the lookahead in route points:
    #
    #     lookahead_pts = 0.9755321901954155 * speed_kmph
    #                     + 1.9152884533402488
    #     lookahead_pts = clip(lookahead_pts, 2.4*ppm, 10.5*ppm)
    #
    # with points_per_meter (ppm) = 10, i.e. in meters:
    #
    #     clip(0.09755 * speed_kmph + 0.19, 2.4, 10.5)
    # --------------------------------------------------------

    def get_angle_lookahead(self):

        speed_kmh = (
            get_speed_ms(
                self.vehicle
            )
            *
            3.6
        )

        points_per_meter = 10.0

        lookahead_pts = (
            0.9755321901954155 * speed_kmh
            +
            1.9152884533402488
        )

        lookahead_pts = float(
            np.clip(
                lookahead_pts,
                2.4 * points_per_meter,
                10.5 * points_per_meter
            )
        )

        return lookahead_pts / points_per_meter

    def get_target_point(self):

        route = self.route_points

        if len(route) < 2:

            return None

        (
            nearest_index,
            nearest_t
        ) = (
            self.find_nearest_segment()
        )

        p1 = route[
            nearest_index,
            :2
        ]

        p2 = route[
            nearest_index + 1,
            :2
        ]

        segment = (
            p2 - p1
        )

        segment_length = float(
            np.linalg.norm(
                segment
            )
        )

        if segment_length <= 1e-9:

            return route[
                nearest_index + 1
            ].copy()

        remaining_on_segment = (
            segment_length
            *
            (
                1.0
                -
                nearest_t
            )
        )

        lookahead = self.get_angle_lookahead()

        if lookahead <= remaining_on_segment:

            ratio = (
                nearest_t
                +
                lookahead
                /
                segment_length
            )

            target = (
                p1
                +
                ratio * segment
            )

            return np.array(
                [
                    target[0],
                    target[1],
                    route[
                        nearest_index,
                        2
                    ]
                ],
                dtype=np.float64
            )

        remaining = (
            lookahead
            -
            remaining_on_segment
        )

        for i in range(
            nearest_index + 1,
            len(route) - 1
        ):

            a = route[
                i,
                :2
            ]

            b = route[
                i + 1,
                :2
            ]

            segment = (
                b - a
            )

            segment_length = float(
                np.linalg.norm(
                    segment
                )
            )

            if segment_length <= 1e-9:
                continue

            if remaining <= segment_length:

                ratio = (
                    remaining
                    /
                    segment_length
                )

                target = (
                    a
                    +
                    ratio * segment
                )

                return np.array(
                    [
                        target[0],
                        target[1],
                        route[
                            i,
                            2
                        ]
                    ],
                    dtype=np.float64
                )

            remaining -= segment_length

        return route[-1].copy()


    # ========================================================
    # AIM POINT (for lateral/steering) — shorter 4-12m lookahead
    # ========================================================

    def get_aim_point(self):

        route = self.route_points

        if len(route) < 2:

            return None

        (
            nearest_index,
            nearest_t
        ) = (
            self.find_nearest_segment()
        )

        p1 = route[
            nearest_index,
            :2
        ]

        p2 = route[
            nearest_index + 1,
            :2
        ]

        segment = (
            p2 - p1
        )

        segment_length = float(
            np.linalg.norm(
                segment
            )
        )

        if segment_length <= 1e-9:

            return route[
                nearest_index + 1
            ].copy()

        remaining_on_segment = (
            segment_length
            *
            (
                1.0
                -
                nearest_t
            )
        )

        lookahead = (
            self.get_lookahead_distance()
        )

        if lookahead <= remaining_on_segment:

            ratio = (
                nearest_t
                +
                lookahead
                /
                segment_length
            )

            target = (
                p1
                +
                ratio * segment
            )

            return np.array(
                [
                    target[0],
                    target[1],
                    route[
                        nearest_index,
                        2
                    ]
                ],
                dtype=np.float64
            )

        remaining = (
            lookahead
            -
            remaining_on_segment
        )

        for i in range(
            nearest_index + 1,
            len(route) - 1
        ):

            a = route[
                i,
                :2
            ]

            b = route[
                i + 1,
                :2
            ]

            segment = (
                b - a
            )

            segment_length = float(
                np.linalg.norm(
                    segment
                )
            )

            if segment_length <= 1e-9:
                continue

            if remaining <= segment_length:

                ratio = (
                    remaining
                    /
                    segment_length
                )

                target = (
                    a
                    +
                    ratio * segment
                )

                return np.array(
                    [
                        target[0],
                        target[1],
                        route[
                            i,
                            2
                        ]
                    ],
                    dtype=np.float64
                )

            remaining -= segment_length

        return route[-1].copy()


    # ========================================================
    # ANGLE
    # ========================================================

    def calculate_angle(self):

        # Use the collector's speed-scaled target point (2.4-10.5m)
        # for angle to match the PDM-Lite data collector's stored angle.
        target_point = self.get_target_point()

        if target_point is None:

            return 0.0

        local_x, local_y = (
            world_to_ego_xy(
                self.vehicle,
                target_point[0],
                target_point[1]
            )
        )

        # If the target somehow ended behind the car,
        # choose the nearest forward route point.
        if local_x <= 0.0:

            best = None
            best_forward_x = float(
                "inf"
            )

            start = self.nearest_index

            end = min(
                len(self.route_points),
                start + 100
            )

            for i in range(
                start,
                end
            ):

                lx, ly = (
                    world_to_ego_xy(
                        self.vehicle,
                        self.route_points[i, 0],
                        self.route_points[i, 1]
                    )
                )

                if lx <= 0.0:
                    continue

                if lx < best_forward_x:

                    best_forward_x = lx

                    best = (
                        self.route_points[i]
                        .copy()
                    )

            if best is not None:

                target_point = best

                local_x, local_y = (
                    world_to_ego_xy(
                        self.vehicle,
                        best[0],
                        best[1]
                    )
                )

        angle = math.atan2(
            local_y,
            local_x
        )

        self.angle_raw = float(
            wrap_angle(angle)
        )

        return self.angle_raw


    # ========================================================
    # LATERAL DISTANCE
    # ========================================================

    def calculate_lateral_distance(self):

        route = self.route_points

        if len(route) < 2:

            return 0.0

        location = (
            self.vehicle
            .get_location()
        )

        ego = np.array(
            [
                location.x,
                location.y
            ],
            dtype=np.float64
        )

        # ----------------------------------------------------
        # Same convention as dataset.py
        # (CarlaDataset.calculate_lateral_distance), so training
        # and inference compute the input identically.
        #
        # Iterate ALL route segments and pick the segment whose
        # SUPPORTING LINE is nearest, measuring the SIGNED
        # perpendicular distance from the ego to that line in
        # the route's right-normal direction:
        #
        #     right  = (-ty, tx)
        #     lateral = (ego - p1) . right
        #
        # Positive -> ego RIGHT of route.
        # Negative -> ego LEFT of route.
        #
        # NOTE: like dataset.py, we DO NOT clamp the projection
        # onto the segment (no t in [0, 1]) and we select by the
        # minimum absolute perpendicular distance to the
        # supporting line -- NOT by distance to the nearest
        # on-segment point and NOT within a window.
        # ----------------------------------------------------

        eps = 1e-9

        best_abs = float(
            "inf"
        )

        best_lateral = 0.0

        for i in range(
            len(route) - 1
        ):

            p1 = route[
                i,
                :2
            ]

            p2 = route[
                i + 1,
                :2
            ]

            segment = (
                p2 - p1
            )

            length_sq = float(
                np.dot(
                    segment,
                    segment
                )
            )

            # Ignore duplicate / invalid points.
            if length_sq <= eps:
                continue

            length = math.sqrt(
                length_sq
            )

            tx = (
                segment[0]
                /
                length
            )

            ty = (
                segment[1]
                /
                length
            )

            # Right-hand normal.
            right_x = -ty
            right_y = tx

            lateral = (
                (ego[0] - p1[0]) * right_x
                +
                (ego[1] - p1[1]) * right_y
            )

            if abs(lateral) < best_abs:

                best_abs = abs(lateral)
                best_lateral = lateral

        return float(
            best_lateral
        )


    # ========================================================
    # SPEED LIMIT
    # ========================================================

    def get_speed_limit(self):

        return get_carla_speed_limit_ms(
            self.vehicle
        )


    # ========================================================
    # COMMANDS
    #
    # Use the route decisions, but do not invent a direct
    # steering command.
    # ========================================================

    def update_commands(self):

        nearest = (
            self.nearest_index
        )

        current = RoadOption.LANEFOLLOW
        next_option = RoadOption.LANEFOLLOW

        end = min(
            len(self.route_options),
            nearest + 80
        )

        maneuver_index = None

        for i in range(
            nearest,
            end
        ):

            option = (
                self.route_options[i]
            )

            if option in (
                RoadOption.LEFT,
                RoadOption.RIGHT,
                RoadOption.CHANGELANELEFT,
                RoadOption.CHANGELANERIGHT
            ):

                maneuver_index = i

                current = option

                break

        if maneuver_index is not None:

            for j in range(
                maneuver_index + 1,
                len(self.route_options)
            ):

                option = (
                    self.route_options[j]
                )

                if option in (
                    RoadOption.LEFT,
                    RoadOption.RIGHT,
                    RoadOption.CHANGELANELEFT,
                    RoadOption.CHANGELANERIGHT
                ):

                    next_option = option

                    break

        self.command = (
            self.option_to_command(
                current
            )
        )

        self.next_command = (
            self.option_to_command(
                next_option
            )
        )


    # ========================================================
    # UPDATE
    # ========================================================

    def update(self):

        self.find_nearest_segment()

        self.aim_point = (
            self.get_aim_point()
        )

        self.angle_raw = (
            self.calculate_angle()
        )

        self.lateral_raw = (
            self.calculate_lateral_distance()
        )

        self.update_commands()

        self.speed_limit = (
            self.get_speed_limit()
        )


# ============================================================
# HAZARD / OBJECT TELEMETRY
#
# model_v15 consumes sixteen telemetry columns.  The first eight
# (speed / command / junction / speed_limit / angle / next_command /
# theta / lateral_distance) come from CarlaRouteState; the remaining
# eight (indices 8..15) are the object / hazard measurements that the
# training pipeline stored for every frame:
#
#   8  obj_dist           normalized distance to the object that is
#                         currently reducing our target speed
#   9  vehicle_hazard     a blocking vehicle ahead in our lane
#   10 light_hazard       a non-green traffic light ahead on the route
#   11 walker_hazard      a walker ahead in our lane
#   12 stop_sign_hazard   a stop sign ahead on the route
#   13 stop_sign_close    stop sign within its trigger volume
#   14 walker_close       walker close enough to force a slowdown
#   15 obj_type           category id of the speed-reducing object:
#                         0 none / 1 vehicle / 2 walker
#                         3 traffic light / 4 stop sign
#
# In training these were written by the PDM-Lite LB2 data collector's
# heavy forecast-based planner (IDM target speeds + OBB collision
# prediction).  We reproduce the same fields with a compact free-space /
# in-lane detector so the model's inputs stay in-distribution, without
# carrying all of that machinery into drive.py.
# ============================================================

# Forward detection radius (m) for vehicle / walker hazards.  This is a
# FIXED window — it must NOT be scaled by the ego's instantaneous speed.
# Mirrors the PDM-Lite free-space / IDM forecast horizon, which flags
# blocking vehicles within a fixed distance regardless of ego speed.
# Kept short (~2-3 car lengths) so the vehicle hazard only triggers
# when the car is genuinely close — avoids "detecting" obstacles that
# are still far away and allows earlier braking.
HAZARD_OBJECT_RADIUS = 8.0

# How far ahead (m) to walk the route looking for a traffic light /
# stop sign, matching the PDM-Lite light_radius.
HAZARD_SIGN_RADIUS = 64.0

# In-lane half-widths (m).  An actor whose ego-local |y| is under these
# is treated as being in our driving lane.
HAZARD_VEHICLE_LANE_HALF = 3.0
HAZARD_WALKER_LANE_HALF = 2.5

# Stop signs stand at the side of the road, offset from lane centre, so
# a wider lateral band decides whether a sign governs our route.
HAZARD_SIGN_LANE_HALF = 8.0


class CarlaHazardDetector:
    """Reproduce the object / hazard telemetry (indices 8..15) that
    model_v15 was trained on, from a light free-space / in-lane check.
    """

    def __init__(self, world, vehicle, route_state):

        self.world = world
        self.vehicle = vehicle
        self.route_state = route_state

        self.obj_distance = None
        self.obj_type = 0
        self.vehicle_hazard = False
        self.light_hazard = False
        self.walker_hazard = False
        self.walker_close = False
        self.stop_sign_hazard = False
        self.stop_sign_close = False

        # Full hazard picture for this tick: one entry per category
        # (the closest object of that category), so multiple hazards
        # are tracked together instead of collapsing to a single one.
        # Each entry: {"category": str, "type": int, "distance": m}.
        self.hazards = []

        # Distance (m) to the closest active hazard of any type, or
        # None when no hazard is present.
        self.closest_hazard_distance = None

    # ----------------------------------------------------
    # Helpers
    # ----------------------------------------------------

    def _forward_distance(self, actor):

        location = actor.get_location()

        local_x, local_y = world_to_ego_xy(
            self.vehicle,
            location.x,
            location.y
        )

        return local_x, local_y

    def _next_traffic_light(self):

        """Return the NEAREST traffic light governing our driving lane
        that is AHEAD of the ego, and its trigger-box distance.
        Returns (None, None) if no qualifying light exists.

        CARLA 0.9.15's Waypoint has no get_traffic_light(); the
        controlling signal is found via world.get_traffic_lights_from_
        waypoint(wp, d) — the same call the PDM-Lite planner makes —
        which returns the lights governing the waypoint's lane within
        d meters ahead.

        We iterate ALL returned lights, compute ego-local forward
        distance, and keep only the closest one that is actually
        ahead (local_x > 0).  This prevents a far-away or behind-ego
        light from hijacking obj_distance when a vehicle is the real
        hazard.
        """

        waypoint = self.world.get_map().get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if waypoint is None:
            return None, None

        lights = self.world.get_traffic_lights_from_waypoint(
            waypoint, HAZARD_SIGN_RADIUS
        )

        if not lights:
            return None, None

        best_light = None
        best_distance = float("inf")

        for light in lights:

            stop_point = light.get_transform().transform(
                light.trigger_volume.location
            )

            local_x, local_y = world_to_ego_xy(
                self.vehicle,
                stop_point.x,
                stop_point.y
            )

            # Only consider lights that are AHEAD of the ego.
            if local_x <= 0.0:
                continue

            distance = stop_point.distance(
                self.vehicle.get_location()
            )

            if distance < best_distance:
                best_distance = distance
                best_light = light

        if best_light is None:
            return None, None

        return best_light, best_distance

    def _next_stop_sign(self):

        """Nearest stop sign standing ahead on our route, or None."""

        best = None
        best_dist = float("inf")

        for actor in self.world.get_actors().filter("*traffic.stop*"):

            location = actor.get_transform().transform(
                actor.trigger_volume.location
            )

            local_x, local_y = world_to_ego_xy(
                self.vehicle,
                location.x,
                location.y
            )

            if local_x <= 0.0 or local_x > HAZARD_SIGN_RADIUS:
                continue

            if abs(local_y) > HAZARD_SIGN_LANE_HALF:
                continue

            dist = math.hypot(local_x, local_y)

            if dist < best_dist:
                best = (actor, dist)
                best_dist = dist

        return best

    # ----------------------------------------------------
    # Per-tick update
    # ----------------------------------------------------

    def update(self):

        self.vehicle_hazard = False
        self.light_hazard = False
        self.walker_hazard = False
        self.walker_close = False
        self.stop_sign_hazard = False
        self.stop_sign_close = False
        self.obj_distance = None
        self.obj_type = 0
        self.hazards = []
        self.closest_hazard_distance = None

        ego_id = self.vehicle.id

        # Fixed detection radius for all hazard categories, so braking
        # cannot shrink the window and drop a detected hazard.
        lookahead = HAZARD_OBJECT_RADIUS

        # Primary (highest-priority) speed-reducing object, used only to
        # fill the model's SINGLE obj_distance / obj_type slots (the model
        # was trained on one speed-reducing object per frame).  All the
        # hazards of the tick are collected in self.hazards below, so
        # nothing is lost.
        #
        # Priority mirrors PDM-Lite's min-target-speed choice: a blocking
        # vehicle is a hard stop, then walker / red light / stop sign.
        blocking = None

        # --- vehicles (closest in-lane vehicle ahead) ---
        closest_vehicle = None

        for actor in self.world.get_actors().filter("*vehicle*"):

            if actor.id == ego_id:
                continue

            local_x, local_y = self._forward_distance(actor)

            if local_x <= 0.0 or local_x > lookahead:
                continue

            if abs(local_y) > HAZARD_VEHICLE_LANE_HALF:
                continue

            self.vehicle_hazard = True

            distance = math.hypot(local_x, local_y)

            # Primary slot: vehicles always outrank other hazard types.
            if blocking is None or 0 < blocking[0] or (0 == blocking[0] and distance < blocking[1]):
                blocking = (0, distance, 1)

            # Full list: keep the closest vehicle of the tick.
            if closest_vehicle is None or distance < closest_vehicle[1]:
                closest_vehicle = (1, distance)

        if closest_vehicle is not None:
            self.hazards.append({
                "category": "vehicle",
                "type": 1,
                "distance": closest_vehicle[1],
            })

        # --- walkers (closest in-lane walker ahead) ---
        closest_walker = None

        for actor in self.world.get_actors().filter("*walker*"):

            local_x, local_y = self._forward_distance(actor)

            if local_x <= 0.0 or local_x > lookahead:
                continue

            if abs(local_y) > HAZARD_WALKER_LANE_HALF:
                continue

            self.walker_hazard = True
            self.walker_close = True

            distance = math.hypot(local_x, local_y)

            if blocking is None or 1 < blocking[0] or (1 == blocking[0] and distance < blocking[1]):
                blocking = (1, distance, 2)

            if closest_walker is None or distance < closest_walker[1]:
                closest_walker = (2, distance)

        if closest_walker is not None:
            self.hazards.append({
                "category": "walker",
                "type": 2,
                "distance": closest_walker[1],
            })

        # --- traffic light (nearest non-green ahead) ---
        light, light_distance = self._next_traffic_light()

        if (
            light is not None
            and light.state != carla.TrafficLightState.Green
        ):

            self.light_hazard = True

            if blocking is None or 2 < blocking[0] or (2 == blocking[0] and light_distance < blocking[1]):
                blocking = (2, light_distance, 3)

            self.hazards.append({
                "category": "traffic_light",
                "type": 3,
                "distance": light_distance,
            })

        # --- stop sign (nearest ahead on route) ---
        stop = self._next_stop_sign()

        if stop is not None:

            self.stop_sign_hazard = True
            self.stop_sign_close = True

            if blocking is None or 3 < blocking[0] or (3 == blocking[0] and stop[1] < blocking[1]):
                blocking = (3, stop[1], 4)

            self.hazards.append({
                "category": "stop_sign",
                "type": 4,
                "distance": stop[1],
            })

        if blocking is not None:
            self.obj_type = blocking[2]
            self.obj_distance = blocking[1]

        # Sort the full list with the closest hazard first, and record
        # the closest of any type for quick reference / display.
        self.hazards.sort(key=lambda h: h["distance"])

        if self.hazards:
            self.closest_hazard_distance = self.hazards[0]["distance"]


# ============================================================
# TELEMETRY
# ============================================================

# Toggle: True = use direct CARLA state (like measurement files), False = use CarlaRouteState
USE_DIRECT_TELEMETRY = True


def build_model_telemetry(
    vehicle,
    world,
    route_state,
    theta,
    hazard
):
    """
    Build 16-dim telemetry for the model, matching the training pipeline.

    Indices 0..7 come from CarlaRouteState (angle/lateral match the data
    collector's route lookahead logic); indices 8..15 are the object /
    hazard measurements from CarlaHazardDetector, normalized exactly like
    dataset.py.
    """

    speed_raw = safe_float(
        get_speed_ms(vehicle),
        "vehicle speed",
        0.0
    )

    # Speed limit from CARLA
    speed_limit_raw = safe_float(
        get_carla_speed_limit_ms(vehicle),
        "CARLA speed limit",
        50.0 / 3.6
    )

    # Junction from current waypoint
    waypoint = world.get_map().get_waypoint(
        vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    junction = float(bool(waypoint.is_junction)) if waypoint is not None else 0.0

    # Command from route_state (uses route decisions)
    command = float(route_state.command)
    next_command = float(route_state.next_command)

    # Angle: use precomputed value from route_state (target_point via
    # the collector's speed-scaled 2.4-10.5m lookahead)
    # This matches how the data collector computes 'angle' in measurements
    angle_raw = safe_float(route_state.angle_raw, "route angle", 0.0)

    # Lateral distance: use precomputed value from route_state
    # This matches how the data collector computes 'lateral' in measurements
    lateral_raw = safe_float(route_state.lateral_raw, "lateral distance", 0.0)

    # ----------------------------------------------------
    # NORMALIZATION (must match dataset.py exactly)
    #
    # Object / hazard columns: object distance uses a 50 m reference
    # clipped to [0, 2] (a missing / far object -> 2.0, like dataset.py's
    # 500 m default); the six hazard flags are 0.0 / 1.0; obj_type is a
    # small category id (0..4) that is embedded, not normalized.
    # ----------------------------------------------------

    obj_distance_raw = (
        500.0
        if hazard.obj_distance is None
        else float(hazard.obj_distance)
    )

    obj_type = int(hazard.obj_type)

    telemetry = np.array(
        [
            # 0 speed: clipped to [0, 2] with 30 m/s reference
            np.clip(speed_raw / 30.0, 0.0, 2.0),

            # 1 command: 1-6 (LEFT, RIGHT, STRAIGHT, LANEFOLLOW, CHANGE_LANE_L/R)
            command,

            # 2 junction: 0.0 or 1.0
            junction,

            # 3 speed_limit: clipped to [0, 2] with 30 m/s reference
            np.clip(speed_limit_raw / 30.0, 0.0, 2.0),

            # 4 angle: MUST match dataset.py exactly.
            # dataset.py reads the raw angle from the measurement file
            # (in radians) and normalizes it as:
            #     np.clip(angle / np.pi, -1.0, 1.0)
            # angle_raw from route_state is already in radians, so apply
            # the same normalization at inference:
            np.clip(angle_raw / math.pi, -1.0, 1.0),

            # 5 next_command: 1-6
            next_command,

            # 6 theta: clipped to [-1, 1] with pi reference
            np.clip(theta / math.pi, -1.0, 1.0),

            # 7 lateral_distance: clipped to [-1, 1] with 2m reference
            np.clip(lateral_raw / 2.0, -1.0, 1.0),

            # 8 obj_dist: clipped to [0, 2] with 50 m reference
            np.clip(obj_distance_raw / 50.0, 0.0, 2.0),

            # 9-14 hazard flags: 0.0 / 1.0
            float(hazard.vehicle_hazard),
            float(hazard.light_hazard),
            float(hazard.walker_hazard),
            float(hazard.stop_sign_hazard),
            float(hazard.stop_sign_close),
            float(hazard.walker_close),

            # 15 obj_type: category id (embedded, not normalized)
            float(obj_type),
        ],
        dtype=np.float32
    )

    if telemetry.shape != (16,):
        raise RuntimeError(f"Telemetry shape mismatch: {telemetry.shape}")

    if not np.all(np.isfinite(telemetry)):
        raise RuntimeError("Telemetry contains NaN/Inf.")

    raw = {
        "speed": speed_raw,
        "speed_limit": speed_limit_raw,
        "angle": angle_raw,
        "theta": theta,
        "lateral": lateral_raw,
        "obj_distance": obj_distance_raw,
        "obj_type": obj_type,
        "hazards": list(getattr(hazard, "hazards", [])),
        "closest_hazard_distance": getattr(hazard, "closest_hazard_distance", None),
        "vehicle_hazard": bool(hazard.vehicle_hazard),
        "light_hazard": bool(hazard.light_hazard),
        "walker_hazard": bool(hazard.walker_hazard),
        "stop_sign_hazard": bool(hazard.stop_sign_hazard),
        "stop_sign_close": bool(hazard.stop_sign_close),
        "walker_close": bool(hazard.walker_close),
        "vehicle_location": [
            vehicle.get_location().x,
            vehicle.get_location().y,
            vehicle.get_location().z
        ],
        "vehicle_yaw": vehicle.get_transform().rotation.yaw,
        "aim_wp": (
            None
            if route_state.aim_point is None
            else route_state.aim_point.copy()
        ),
    }

    return telemetry, raw


# ============================================================
# MODEL
# ============================================================

def load_model():

    print()
    print("=" * 82)
    print("LOADING MODEL")
    print("=" * 82)

    if not os.path.isfile(
        MODEL_FILE
    ):

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_FILE}"
        )

    model = AutonomousDriver()

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model" in checkpoint:
            state_dict = checkpoint[
                "model"
            ]

        else:
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[7:]

        cleaned[key] = value

    model.load_state_dict(
        cleaned,
        strict=True
    )

    model.to(
        DEVICE
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    print(
        "Device:",
        DEVICE
    )

    print(
        "Telemetry:",
        MODEL_TELEMETRY
    )

    print(
        "Checkpoint:",
        MODEL_FILE
    )

    return model


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def run_model(
    model,
    rgb,
    telemetry
):

    # Crop the top 30% (sky / buildings / trees) to match the
    # dataset's preprocessing, so the model sees the same road-only
    # view at inference as it was trained on.
    crop_h = int(
        rgb.shape[0]
        * IMAGE_CROP_RATIO
    )
    rgb = rgb[crop_h:, :]

    image_tensor = (
        IMAGE_TRANSFORM(
            rgb
        )
        .unsqueeze(0)
        .to(
            DEVICE,
            non_blocking=True
        )
    )

    telemetry_tensor = (
        torch
        .from_numpy(
            telemetry
        )
        .float()
        .unsqueeze(0)
        .to(
            DEVICE,
            non_blocking=True
        )
    )

    # model returns a tuple:
    #   steering    : [B, 1]  (Tanh    -> [-1, 1])
    #   throttle    : [B, 1]  (Sigmoid -> [ 0, 1])
    #   brake_logits: [B, 2]  (2-class classification logits)
    steering, throttle, brake_logits = model(
        image_tensor,
        telemetry_tensor
    )

    steering = float(
        np.clip(
            steering.item(),
            -1.0,
            1.0
        )
    )

    throttle = float(
        np.clip(
            throttle.item(),
            0.0,
            1.0
        )
    )

    # Brake head outputs 2-class logits (trained with cross-entropy
    # against a binary control_brake target).  Softmax turns them into
    # class probabilities; class 1 is the brake probability — a continuous
    # value in [0, 1], passed straight through to CARLA as the brake.
    brake_probs = torch.softmax(
        brake_logits,
        dim=1
    )

    brake_probability = float(
        np.clip(
            brake_probs[0, 1].item(),
            0.0,
            1.0
        )
    )

    return (
        steering,
        throttle,
        brake_probability
    )


# ============================================================
# CONTROL
# ============================================================

def apply_control(
    vehicle,
    steering,
    throttle,
    brake_probability
):

    global _last_steering, _last_steering_smoothed

    speed_ms = get_speed_ms(vehicle)
    speed_kmh = speed_ms * 3.6

    # --- Steering: single EMA smoothing ---
    # At very low speed (standstill / just started), use less smoothing so
    # the car can actually begin turning.  At higher speed, smooth more to
    # avoid oscillation.
    if speed_kmh < 5.0:
        alpha = 0.25   # fast response at low speed
    elif speed_kmh < 20.0:
        alpha = 0.35   # moderate
    else:
        alpha = STEER_SMOOTHING  # smooth at cruising speed

    steering_applied = float(np.clip(
        alpha * steering + (1.0 - alpha) * _last_steering_smoothed,
        -1.0,
        1.0,
    ))

    _last_steering = steering
    _last_steering_smoothed = steering_applied

    # --- Brake / throttle arbitration ---
    # Brake head is a binary classifier, so use the probability only
    # to decide whether braking is active. Never send throttle and brake
    # simultaneously.
    if brake_probability >= BRAKE_THRESHOLD:
        brake = 1.0
        throttle = 0.0
    else:
        brake = 0.0
        throttle = float(np.clip(throttle, 0.0, MAX_THROTTLE))

    control = carla.VehicleControl(
        steer=steering_applied,
        throttle=throttle,
        brake=brake,
        hand_brake=False,
        reverse=False,
        manual_gear_shift=False
    )

    vehicle.apply_control(control)

    return (control.steer, control.throttle, control.brake)


# ============================================================
# CAMERA DISPLAY
# ============================================================

def show_camera(
    rgb,
    telemetry,
    raw,
    steering,
    throttle,
    brake_probability
):

    if not SHOW_CAMERA:
        return

    frame = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    lines = [
        f"Speed: {raw['speed']:.2f} m/s",

        (
            f"Command: "
            f"{int(telemetry[1])} "
            f"{command_name(telemetry[1])}"
        ),

        (
            f"Next: "
            f"{int(telemetry[5])} "
            f"{command_name(telemetry[5])}"
        ),

        (
            f"Vehicle yaw: "
            f"{raw['vehicle_yaw']:+.2f}"
        ),

        (
            f"Angle: "
            f"{raw['angle']:+.3f}"
        ),

        (
            f"Theta: "
            f"{raw['theta']:+.3f}"
        ),

        (
            f"Lateral: "
            f"{raw['lateral']:+.3f} m"
        ),

        (
            f"Steer: "
            f"{steering:+.3f}"
        ),

        (
            f"Throttle: "
            f"{throttle:.3f}"
        ),

        (
            f"Brake P: "
            f"{brake_probability:.3f}"
        ),
    ]

    y = 25

    for text in lines:

        cv2.putText(
            frame,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        y += 25

    h, w = frame.shape[:2]

    cv2.line(
        frame,
        (w // 2, h - 70),
        (w // 2, h - 20),
        (0, 255, 255),
        2
    )

    indicator_x = int(
        (w / 2)
        +
        steering
        *
        (w * 0.25)
    )

    cv2.line(
        frame,
        (w // 2, h - 45),
        (indicator_x, h - 45),
        (0, 255, 0),
        4
    )

    try:
        cv2.imshow(
            "PDM-Lite Neural Driver - Camera",
            frame
        )
    except cv2.error:
        # Headless / no-display environment: don't crash the driver.
        pass


# ============================================================
# DEBUG
# ============================================================

def print_debug(
    telemetry,
    raw,
    vehicle,
    compass,
    theta,
    steering,
    throttle,
    brake_probability
):

    print(
        "\033[2J\033[H",
        end=""
    )

    print("=" * 82)

    print(
        "                 PDM-LITE NEURAL DRIVER V15"
    )

    print("=" * 82)

    print(
        f"Vehicle location: "
        f"("
        f"{raw['vehicle_location'][0]:+.3f}, "
        f"{raw['vehicle_location'][1]:+.3f}, "
        f"{raw['vehicle_location'][2]:+.3f}"
        f")"
    )

    print(
        f"Vehicle yaw     : "
        f"{raw['vehicle_yaw']:+.6f} deg"
    )

    print(
        f"IMU compass     : "
        f"{compass:+.9f} rad"
    )

    print(
        f"PDM theta raw   : "
        f"{theta:+.9f} rad"
    )

    print("-" * 82)

    print(
        "RAW CARLA ROUTE"
    )

    print(
        f"Speed raw       : "
        f"{raw['speed']:+.6f} m/s"
    )

    print(
        f"Speed limit raw : "
        f"{raw['speed_limit']:+.6f} m/s"
    )

    print(
        f"Angle raw       : "
        f"{raw['angle']:+.6f} rad"
    )

    print(
        f"Lateral raw     : "
        f"{raw['lateral']:+.6f} m"
    )

    print(
        f"Aim WP          : "
        f"{raw['aim_wp']}"
    )

    print("-" * 82)

    print(
        "MODEL INPUT — 16 TELEMETRY"
    )

    labels = [
        "speed",
        "command",
        "junction",
        "speed_limit",
        "angle",
        "next_command",
        "theta",
        "lateral_distance",
        "obj_distance",
        "vehicle_hazard",
        "light_hazard",
        "walker_hazard",
        "stop_sign_hazard",
        "stop_sign_close",
        "walker_close",
        "obj_type",
    ]

    for i, label in enumerate(labels):

        if i in (
            1,
            5
        ):

            print(
                f"{label:<18}: "
                f"{int(telemetry[i])} "
                f"{command_name(telemetry[i])}"
            )

        elif i == 15:

            _obj_names = {0: "none", 1: "vehicle", 2: "walker", 3: "traffic_light", 4: "stop_sign"}
            _obj_id = int(telemetry[i])
            print(
                f"{label:<18}: "
                f"{_obj_id} ({_obj_names.get(_obj_id, '?')})"
            )

        else:

            print(
                f"{label:<18}: "
                f"{telemetry[i]:+.6f}"
            )

    print("-" * 82)

    print(
        "HAZARDS (all active this tick)"
    )

    hazards = raw.get("hazards", [])

    if hazards:

        closest = raw.get("closest_hazard_distance")

        for item in hazards:

            marker = "  <-- closest" if closest is not None and abs(item["distance"] - closest) < 1e-9 else ""

            print(
                f"  {item['category']:<14}: "
                f"{item['distance']:+.2f} m"
                f"{marker}"
            )

        print(
            f"  Closest hazard  : "
            f"{closest:+.2f} m" if closest is not None else "  Closest hazard  : (none)"
        )

    else:

        print(
            "  (none)"
        )

    print("-" * 82)

    print(
        "MODEL OUTPUT"
    )

    print(
        f"Steering         : "
        f"{steering:+.6f}"
    )

    print(
        f"Throttle         : "
        f"{throttle:+.6f}"
    )

    print(
        f"Brake probability: "
        f"{brake_probability:+.6f}"
    )

    print("=" * 82)


# ============================================================
# TOWN PROBE
# ============================================================

def test_towns(client, towns=None):
    """
    Probe each town by loading it and immediately unloading.
    Returns (working, failed) lists so --test-towns can show
    which maps are safe to use.
    """
    if towns is None:
        towns = TOWNS

    working = []
    failed = []

    print()
    print("=" * 82)
    print("PROBING TOWNS")
    print("=" * 82)

    for town in towns:
        print(f"  Testing {town}...", end=" ", flush=True)
        try:
            # Load the town (CARLA handles the heavy lifting).
            client.load_world(town)
            # Give the engine a moment to crash if it will.
            time.sleep(1.0)
            # If we got here, mark it working.
            working.append(town)
            print("OK")
        except Exception as exc:
            failed.append((town, str(exc)))
            print(f"FAILED ({exc})")

    print()
    print("=" * 82)
    print("RESULTS")
    print("=" * 82)

    if working:
        print(f"  Working : {working}")

    if failed:
        print(f"  Failed  : {[t for t, _ in failed]}")

    # Persist confirmed towns so the next run can skip known-bad ones.
    try:
        with open(CONFIRMED_FILE, "w") as fh:
            for t in working:
                fh.write(t + "\n")
    except Exception:
        pass

    return working, failed


# ============================================================
# ROUTE RUNNER
# ============================================================

def run_single_route(
    client,
    town,
    model,
    route_index=0,
    num_routes=1,
):
    """
    Drive one complete route in the given town.
    Returns True if the destination was reached, False otherwise.
    """
    world = None
    vehicle = None
    camera = None
    imu = None
    original_settings = None

    try:
        world = load_navigation_town(client, town)

        # Sync
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        os.chdir(PDM_ROOT)

        vehicle = spawn_vehicle(world)

        destination, global_route = choose_destination(
            world, vehicle
        )

        # Drive from the actual random spawn point — no re-anchoring.
        # The old code teleported the vehicle to global_route[0][0]'s
        # road waypoint, which converged to the same spot across runs
        # because the GlobalRoutePlanner snaps to lane centers.

        # Populate the world with NPC traffic + scenery
        # (autopilot cars start driving immediately via TrafficManager).
        if TRAFFIC_VEHICLES > 0:
            spawn_traffic(
                world,
                client,
                vehicle,
                count=TRAFFIC_VEHICLES,
                tm_port=TRAFFIC_PORT,
            )

        print()
        print(f"--- Route {route_index + 1}/{num_routes} in {town} ---")
        print(f"  Destination : {destination.location}")
        print(f"  Route points: {len(global_route)}")
        print(
            "  Decisions   :",
            route_turn_summary(global_route) or ["STRAIGHT"],
        )

        # Camera
        library = world.get_blueprint_library()

        camera_bp = library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        camera_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        camera_bp.set_attribute("fov", str(CAMERA_FOV))
        camera_bp.set_attribute(
            "sensor_tick", str(FIXED_DELTA_SECONDS)
        )

        camera = world.spawn_actor(
            camera_bp,
            carla.Transform(
                carla.Location(x=CAMERA_X, y=CAMERA_Y, z=CAMERA_Z),
                carla.Rotation(
                    roll=CAMERA_ROLL,
                    pitch=CAMERA_PITCH,
                    yaw=CAMERA_YAW,
                ),
            ),
            attach_to=vehicle,
        )
        camera.listen(camera_callback)

        # IMU
        imu_bp = library.find("sensor.other.imu")
        imu_bp.set_attribute(
            "sensor_tick", str(IMU_SENSOR_TICK)
        )

        imu = world.spawn_actor(
            imu_bp,
            carla.Transform(
                carla.Location(x=0.0, y=0.0, z=0.0)
            ),
            attach_to=vehicle,
        )
        imu.listen(imu_callback)

        # Warm-up
        WARMUP_SECONDS = 5.0
        WARMUP_TICKS = int(WARMUP_SECONDS / FIXED_DELTA_SECONDS)

        print(
            f"Warming up sensors for "
            f"{WARMUP_SECONDS:.0f}s..."
        )

        for _ in range(WARMUP_TICKS):
            world.tick()
            vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0, steer=0.0, brake=1.0
                )
            )
            while not camera_queue.empty():
                camera_queue.get_nowait()
            while not imu_queue.empty():
                imu_queue.get_nowait()

        print("Warm-up complete.")

        # Prime sensors
        frame = world.tick()
        get_sensor_frame_data(camera_queue, frame)
        get_sensor_frame_data(imu_queue, frame)

        # Ensure route starts at vehicle
        global_route = ensure_route_starts_at_vehicle(
            vehicle, global_route
        )

        route_state = CarlaRouteState(
            world, vehicle, global_route
        )

        hazard = CarlaHazardDetector(
            world, vehicle, route_state
        )

        vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0, steer=0.0, brake=1.0
            )
        )
        world.tick()

        # ---- DRIVING LOOP ----
        print()
        print("=" * 82)
        print(
            f"DRIVING  —  Route {route_index + 1}/{num_routes}"
            f"  Town {town}"
        )
        print("=" * 82)
        print("Press Q to stop.")
        print("Destination:", destination.location)

        while True:
            frame = world.tick()

            camera_data = get_sensor_frame_data(
                camera_queue, frame
            )
            imu_data = get_sensor_frame_data(
                imu_queue, frame
            )

            rgb = image_to_rgb(camera_data)

            compass, theta = get_pdm_theta(imu_data)

            route_state.update()

            hazard.update()

            # Destination reached?
            dist_to_dest = vehicle.get_location().distance(
                destination.location
            )

            if dist_to_dest < DESTINATION_REACHED_DISTANCE:
                vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.0, steer=0.0, brake=1.0
                    )
                )
                print()
                print("Destination reached!")
                return True

            telemetry, raw = build_model_telemetry(
                vehicle, world, route_state, theta, hazard
            )

            # Sanity checks
            if not (-1.0 <= float(telemetry[4]) <= 1.0):
                raise RuntimeError(
                    f"Angle outside range: {telemetry[4]}"
                )
            if not (-1.0 <= float(telemetry[6]) <= 1.0):
                raise RuntimeError(
                    f"Theta outside range: {telemetry[6]}"
                )
            if not (-1.0 <= float(telemetry[7]) <= 1.0):
                raise RuntimeError(
                    f"Lateral outside range: {telemetry[7]}"
                )
            if not np.all(np.isfinite(telemetry)):
                raise RuntimeError(
                    "Telemetry contains NaN/Inf."
                )

            steering, throttle, brake_p = run_model(
                model, rgb, telemetry
            )

            applied_steer, applied_throttle, applied_brake = (
                apply_control(vehicle, steering, throttle, brake_p)
            )

            if PRINT_TELEMETRY:
                print_debug(
                    telemetry,
                    raw,
                    vehicle,
                    compass,
                    theta,
                    applied_steer,
                    applied_throttle,
                    brake_p,
                )

            show_camera(
                rgb,
                telemetry,
                raw,
                applied_steer,
                applied_throttle,
                brake_p,
            )

            if SHOW_CAMERA:
                try:
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error:
                    key = -1

                if key == ord("q"):
                    print("\nQ pressed.")
                    return False

            if PRINT_TELEMETRY:
                print(
                    f"Distance to destination: "
                    f"{dist_to_dest:.2f} m"
                )

    finally:
        # Cleanup
        if vehicle is not None:
            try:
                vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.0, steer=0.0, brake=1.0
                    )
                )
            except Exception:
                pass

        for sensor in (camera, imu):
            if sensor is not None:
                try:
                    sensor.stop()
                except Exception:
                    pass
                try:
                    sensor.destroy()
                except Exception:
                    pass

        if vehicle is not None:
            try:
                vehicle.destroy()
            except Exception:
                pass

        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception:
                pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# ============================================================
# MAIN
# ============================================================

def main():

    global MODEL_FILE, BRAKE_THRESHOLD, PRINT_TELEMETRY, \
        TRAFFIC_VEHICLES, TRAFFIC_PORT
    global _last_steering, _last_steering_smoothed

    parser = argparse.ArgumentParser(
        description=(
            "PDM-Lite Neural Driver V15\n\n"
            "Examples:\n"
            "  python drive.py --test-towns\n"
            "    Probe which CARLA towns load OK.\n\n"
            "  python drive.py --town Town04\n"
            "    Drive a single route in Town04.\n\n"
            "  python drive.py --num-routes 5\n"
            "    Drive 5 random routes in random towns.\n\n"
            "  python drive.py --num-routes 3 --towns Town01 Town05\n"
            "    Cycle through Town01 and Town05 for 3 routes.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--town",
        choices=TOWNS,
        default=None,
        help=(
            "CARLA town. "
            "If omitted, selected randomly."
        )
    )

    parser.add_argument(
        "--model",
        default=MODEL_FILE,
        help=(
            "Path to model checkpoint (.pth). "
            f"Default: {MODEL_FILE}"
        )
    )

    parser.add_argument(
        "--brake-threshold",
        type=float,
        default=BRAKE_THRESHOLD,
        help=(
            "Brake when braking probability exceeds "
            "this threshold [0, 1]. "
            f"Default: {BRAKE_THRESHOLD}"
        )
    )

    parser.add_argument(
        "--traffic",
        type=int,
        default=TRAFFIC_VEHICLES,
        help=(
            "Number of autopilot NPC vehicles to spawn in "
            "the world before driving (0 disables). "
            f"Default: {TRAFFIC_VEHICLES}"
        )
    )

    parser.add_argument(
        "--traffic-port",
        type=int,
        default=TRAFFIC_PORT,
        help=(
            "TrafficManager port that drives the NPC vehicles. "
            f"Default: {TRAFFIC_PORT}"
        )
    )

    parser.add_argument(
        "--num-routes",
        type=int,
        default=1,
        help=(
            "Number of routes to drive in sequence. "
            "After each route completes (or crashes), "
            "the next one starts automatically. "
            "Default: 1"
        )
    )

    parser.add_argument(
        "--test-towns",
        action="store_true",
        help=(
            "Probe each CARLA town to see which ones "
            "load successfully, then exit. Useful for "
            "diagnosing segfaults on other maps."
        )
    )

    parser.add_argument(
        "--towns",
        nargs="+",
        default=None,
        help=(
            "Explicit list of towns to try when "
            "--num-routes > 1. Example: "
            "--towns Town01 Town04 Town05. "
            "If omitted, uses the default TOWNS list."
        )
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Suppress per-frame terminal telemetry. "
            "Route info and destination messages still print."
        )
    )

    args = parser.parse_args()

    MODEL_FILE = args.model
    BRAKE_THRESHOLD = float(
        np.clip(
            args.brake_threshold,
            0.0,
            1.0
        )
    )

    TRAFFIC_VEHICLES = max(0, args.traffic)
    TRAFFIC_PORT = args.traffic_port

    num_routes = max(1, args.num_routes)

    if args.quiet:
        PRINT_TELEMETRY = False

    original_cwd = os.getcwd()

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 82)

    print(
        "                 PDM-LITE NEURAL DRIVER V15"
    )

    print("=" * 82)

    print(
        "Device     :",
        DEVICE
    )

    print(
        "Model      :",
        MODEL_FILE
    )

    print(
        "Input      : "
        f"{MODEL_WIDTH} x "
        f"{MODEL_HEIGHT}"
    )

    print(
        "Telemetry  :",
        MODEL_TELEMETRY
    )

    print(
        "Camera     : "
        "1024 x 512 / FOV 110"
    )

    print(
        "Navigation : "
        "direct CARLA GlobalRoutePlanner"
    )

    print(
        "Routes     :",
        num_routes,
    )

    if args.town:
        print(
            "Town       :",
            args.town,
        )
    else:
        print(
            "Town       : (random)"
        )

    print("=" * 82)

    # ========================================================
    # CONNECT TO CARLA (once, reuse across routes)
    # ========================================================

    client = carla.Client(
        CARLA_HOST,
        CARLA_PORT
    )

    client.set_timeout(60.0)

    print(
        "CARLA client:",
        client.get_client_version()
    )

    print(
        "CARLA server:",
        client.get_server_version()
    )

    # ========================================================
    # --test-towns: probe and exit
    # ========================================================

    if args.test_towns:
        test_towns(client, args.towns or TOWNS)
        return

    # ========================================================
    # BUILD TOWN LIST
    # ========================================================

    # If the user supplied --town, use exactly that town.
    # Otherwise, try confirmed towns first, then all remaining.
    if args.town:
        town_list = [args.town] * num_routes
    elif args.towns:
        town_list = [
            args.towns[i % len(args.towns)]
            for i in range(num_routes)
        ]
    else:
        # Try loading previously confirmed towns.
        confirmed = []
        if os.path.isfile(CONFIRMED_FILE):
            try:
                with open(CONFIRMED_FILE) as fh:
                    confirmed = [
                        line.strip()
                        for line in fh
                        if line.strip()
                        and line.strip() in TOWNS
                    ]
            except Exception:
                pass

        pool = (
            confirmed
            if len(confirmed) >= 2
            else TOWNS
        )

        town_list = [
            random.choice(pool)
            for _ in range(num_routes)
        ]

    # ========================================================
    # LOAD MODEL (once)
    # ========================================================

    model = load_model()

    # ========================================================
    # DRIVE ROUTES
    # ========================================================

    results = []

    for route_idx in range(num_routes):
        selected_town = town_list[route_idx]

        print()
        print("#" * 82)
        print(
            f"#  ROUTE {route_idx + 1}/{num_routes}"
            f"   TOWN: {selected_town}"
        )
        print("#" * 82)

        try:
            reached = run_single_route(
                client,
                selected_town,
                model,
                route_index=route_idx,
                num_routes=num_routes,
            )
            results.append(
                (selected_town, "reached" if reached else "quit")
            )

        except Exception as exc:
            results.append((selected_town, f"error: {exc}"))

            print()
            print(
                f"Route {route_idx + 1} failed: "
                f"{type(exc).__name__}: {exc}"
            )

            # If a town segfaults, try to move to a
            # different one for the next route.
            if route_idx + 1 < num_routes:
                fallback = random.choice(
                    [t for t in TOWNS if t != selected_town]
                )
                town_list[route_idx + 1] = fallback
                print(
                    f"  Falling back to {fallback}"
                    f" for next route."
                )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 82)
    print("SESSION SUMMARY")
    print("=" * 82)

    for idx, (town, outcome) in enumerate(results):
        print(
            f"  Route {idx + 1}: "
            f"{town:<12}  {outcome}"
        )

    print("=" * 82)

    # ========================================================
    # CLEANUP
    # ========================================================

    try:
        os.chdir(original_cwd)
    except Exception:
        pass

    print("Driver stopped.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
