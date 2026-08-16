import os
import sys
import math
import time
import queue
import random
import traceback
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
    "model_v9.pth"
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

MODEL_WIDTH = 400
MODEL_HEIGHT = 200
MODEL_TELEMETRY = 11


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
# CONTROL
# ============================================================

BRAKE_THRESHOLD = 0.50
MAX_THROTTLE = 0.40

STEER_SMOOTHING = 0.35
MAX_STEER_STEP = 0.12


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
]


# ============================================================
# ROUTE SELECTION
# ============================================================

MIN_ROUTE_DISTANCE = 180.0
MAX_ROUTE_DISTANCE = 1500.0
DESTINATION_ATTEMPTS = 150
REQUIRE_TURN = True


# ============================================================
# COMMAND NAMES
# ============================================================

COMMAND_NAMES = {
    1: "LEFT",
    2: "RIGHT",
    3: "STRAIGHT",
    4: "LANEFOLLOW",
    5: "CHANGE_LANE_LEFT",
    6: "CHANGE_LANE_RIGHT",
}


# ============================================================
# IMAGE
# ============================================================

IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(
        (
            MODEL_HEIGHT,
            MODEL_WIDTH
        )
    ),
    transforms.ToTensor(),
])


# ============================================================
# SENSOR QUEUES
# ============================================================

camera_queue = queue.Queue()
imu_queue = queue.Queue()


# ============================================================
# STEERING STATE
# ============================================================

_last_steering = 0.0


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

        if data.frame == frame:
            return data

        raise RuntimeError(
            f"Sensor frame mismatch: "
            f"wanted {frame}, "
            f"got {data.frame}."
        )


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

    for transform in spawn_points:

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

        # Search whole route.
        # This is safer than assuming the planner's index
        # is always synchronized.
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


    # ========================================================
    # AIM POINT
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

        self.aim_point = (
            self.get_aim_point()
        )

        if self.aim_point is None:

            return 0.0

        local_x, local_y = (
            world_to_ego_xy(
                self.vehicle,
                self.aim_point[0],
                self.aim_point[1]
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

                self.aim_point = best

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

        (
            nearest_index,
            _
        ) = (
            self.find_nearest_segment()
        )

        start = max(
            0,
            nearest_index - 5
        )

        end = min(
            len(route) - 1,
            nearest_index + 10
        )

        best_distance_sq = float(
            "inf"
        )

        best_lateral = 0.0

        for i in range(
            start,
            end
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
                delta[0] * right_x
                +
                delta[1] * right_y
            )

            if distance_sq < best_distance_sq:

                best_distance_sq = (
                    distance_sq
                )

                best_lateral = (
                    lateral
                )

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
# OBJECT INPUT
# ============================================================

def get_speed_reduction_object():

    return (
        0,
        -1,
        0.0
    )


# ============================================================
# TELEMETRY
# ============================================================

def build_model_telemetry(
    vehicle,
    world,
    route_state,
    theta
):

    speed_raw = (
        get_speed_ms(
            vehicle
        )
    )

    speed_limit_raw = safe_float(
        route_state.speed_limit,
        "speed limit",
        fallback=(
            get_carla_speed_limit_ms(
                vehicle
            )
        )
    )

    angle_raw = safe_float(
        route_state.angle_raw,
        "route angle"
    )

    lateral_raw = safe_float(
        route_state.lateral_raw,
        "lateral distance"
    )

    waypoint = (
        world
        .get_map()
        .get_waypoint(
            vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
    )

    junction = (
        float(
            bool(
                waypoint.is_junction
            )
        )
        if waypoint is not None
        else 0.0
    )

    (
        object_type,
        object_id,
        object_distance_raw
    ) = get_speed_reduction_object()

    telemetry = np.array(
        [

            # 0 speed
            np.clip(
                speed_raw / 30.0,
                0.0,
                2.0
            ),

            # 1 command
            float(
                route_state.command
            ),

            # 2 junction
            junction,

            # 3 object type
            float(
                object_type
            ),

            # 4 object id
            float(
                object_id
            ),

            # 5 object distance
            np.clip(
                object_distance_raw / 50.0,
                0.0,
                1.0
            ),

            # 6 speed limit
            np.clip(
                speed_limit_raw / 30.0,
                0.0,
                2.0
            ),

            # 7 angle
            np.clip(
                angle_raw / math.pi,
                -1.0,
                1.0
            ),

            # 8 next command
            float(
                route_state.next_command
            ),

            # 9 theta
            np.clip(
                theta / math.pi,
                -1.0,
                1.0
            ),

            # 10 lateral
            np.clip(
                lateral_raw / 2.0,
                -1.0,
                1.0
            ),
        ],
        dtype=np.float32
    )

    if telemetry.shape != (11,):

        raise RuntimeError(
            f"Telemetry shape mismatch: "
            f"{telemetry.shape}"
        )

    if not np.all(
        np.isfinite(
            telemetry
        )
    ):

        raise RuntimeError(
            "Telemetry contains NaN/Inf."
        )

    raw = {
        "speed":
            speed_raw,

        "speed_limit":
            speed_limit_raw,

        "angle":
            angle_raw,

        "theta":
            theta,

        "lateral":
            lateral_raw,

        "vehicle_location":
            [
                vehicle.get_location().x,
                vehicle.get_location().y,
                vehicle.get_location().z
            ],

        "vehicle_yaw":
            vehicle.get_transform().rotation.yaw,

        "aim_wp":
            (
                None
                if route_state.aim_point is None
                else route_state.aim_point.copy()
            ),
    }

    return (
        telemetry,
        raw
    )


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

    output = model(
        image_tensor,
        telemetry_tensor
    )

    if output.shape != (1, 3):

        raise RuntimeError(
            f"Model output shape "
            f"{output.shape}; "
            f"expected (1,3)."
        )

    steering = float(
        np.clip(
            output[0, 0].item(),
            -1.0,
            1.0
        )
    )

    throttle = float(
        np.clip(
            output[0, 1].item(),
            0.0,
            MAX_THROTTLE
        )
    )

    brake_logit = float(
        output[0, 2].item()
    )

    brake_probability = float(
        torch
        .sigmoid(
            output[0, 2]
        )
        .item()
    )

    return (
        steering,
        throttle,
        brake_logit,
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

    global _last_steering

    steering = float(
        np.clip(
            steering,
            -1.0,
            1.0
        )
    )

    throttle = float(
        np.clip(
            throttle,
            0.0,
            MAX_THROTTLE
        )
    )

    delta = (
        steering
        -
        _last_steering
    )

    delta = float(
        np.clip(
            delta,
            -MAX_STEER_STEP,
            MAX_STEER_STEP
        )
    )

    limited = (
        _last_steering
        +
        delta
    )

    steering_applied = (
        STEER_SMOOTHING
        * limited
        +
        (
            1.0
            -
            STEER_SMOOTHING
        )
        *
        _last_steering
    )

    steering_applied = float(
        np.clip(
            steering_applied,
            -1.0,
            1.0
        )
    )

    _last_steering = (
        steering_applied
    )

    brake = (
        brake_probability
        >= BRAKE_THRESHOLD
    )

    if brake:
        throttle = 0.0

    control = carla.VehicleControl(
        steer=steering_applied,
        throttle=throttle,
        brake=(
            1.0
            if brake
            else 0.0
        ),
        hand_brake=False,
        reverse=False,
        manual_gear_shift=False
    )

    vehicle.apply_control(
        control
    )

    return (
        control.steer,
        control.throttle,
        control.brake
    )


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
            f"{int(telemetry[8])} "
            f"{command_name(telemetry[8])}"
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

    cv2.imshow(
        "PDM-Lite Neural Driver - Camera",
        frame
    )


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
    brake_logit,
    brake_probability
):

    print(
        "\033[2J\033[H",
        end=""
    )

    print("=" * 82)

    print(
        "                 PDM-LITE NEURAL DRIVER V9"
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
        "MODEL INPUT — 11 TELEMETRY"
    )

    labels = [
        "speed",
        "command",
        "junction",
        "object_type",
        "object_id",
        "object_distance",
        "speed_limit",
        "angle",
        "next_command",
        "theta",
        "lateral_distance",
    ]

    for i, label in enumerate(labels):

        if i in (
            1,
            8
        ):

            print(
                f"{label:<18}: "
                f"{int(telemetry[i])} "
                f"{command_name(telemetry[i])}"
            )

        else:

            print(
                f"{label:<18}: "
                f"{telemetry[i]:+.6f}"
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
        f"Brake logit      : "
        f"{brake_logit:+.6f}"
    )

    print(
        f"Brake probability: "
        f"{brake_probability:+.6f}"
    )

    print("=" * 82)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "PDM-Lite Neural Driver V9"
        )
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

    args = parser.parse_args()

    selected_town = (
        args.town
        if args.town is not None
        else random.choice(TOWNS)
    )

    client = None
    world = None
    vehicle = None
    camera = None
    imu = None

    original_settings = None
    original_cwd = os.getcwd()

    try:

        # ====================================================
        # HEADER
        # ====================================================

        print()
        print("=" * 82)

        print(
            "                 PDM-LITE NEURAL DRIVER V9"
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
            "Camera pos : "
            "(-1.5, 0.0, 2.0)"
        )

        print(
            "Camera rot : "
            "(roll 0, pitch 0, yaw 0)"
        )

        print(
            "Navigation : "
            "direct CARLA GlobalRoutePlanner"
        )

        print(
            "Route      : "
            "normalized Waypoint -> Transform"
        )

        print(
            "Angle      : "
            "current vehicle yaw -> route lookahead"
        )

        print(
            "Theta      : "
            "IMU -> preprocess_compass()"
        )

        print(
            "Lateral    : "
            "direct route signed distance / 2"
        )

        print("=" * 82)

        # ====================================================
        # CARLA
        # ====================================================

        client = carla.Client(
            CARLA_HOST,
            CARLA_PORT
        )

        client.set_timeout(
            10.0
        )

        world = (
            load_navigation_town(
                client,
                selected_town
            )
        )

        print(
            "CARLA client:",
            client.get_client_version()
        )

        print(
            "CARLA server:",
            client.get_server_version()
        )

        print(
            "Active map:",
            world.get_map().name
        )

        # ====================================================
        # SYNC
        # ====================================================

        original_settings = (
            world.get_settings()
        )

        settings = (
            world.get_settings()
        )

        settings.synchronous_mode = True

        settings.fixed_delta_seconds = (
            FIXED_DELTA_SECONDS
        )

        settings.no_rendering_mode = False

        world.apply_settings(
            settings
        )

        print(
            "Synchronous mode: ON"
        )

        print(
            "Fixed delta:",
            FIXED_DELTA_SECONDS
        )

        # ====================================================
        # PDM WORKING DIRECTORY
        # ====================================================

        os.chdir(
            PDM_ROOT
        )

        print(
            "PDM-Lite working directory:",
            os.getcwd()
        )

        # ====================================================
        # VEHICLE
        # ====================================================

        vehicle = spawn_vehicle(
            world
        )

        # ====================================================
        # DESTINATION
        # ====================================================

        (
            destination,
            global_route
        ) = choose_destination(
            world,
            vehicle
        )

        print()

        print(
            "Selected destination:",
            destination.location
        )

        print(
            "Route points:",
            len(global_route)
        )

        print(
            "Route decisions:",
            route_turn_summary(
                global_route
            )
            or
            ["STRAIGHT"]
        )

        # ====================================================
        # MODEL
        # ====================================================

        model = load_model()

        library = (
            world
            .get_blueprint_library()
        )

        # ====================================================
        # CAMERA
        #
        # EXACTLY UNCHANGED
        # ====================================================

        camera_bp = (
            library.find(
                "sensor.camera.rgb"
            )
        )

        camera_bp.set_attribute(
            "image_size_x",
            str(CAMERA_WIDTH)
        )

        camera_bp.set_attribute(
            "image_size_y",
            str(CAMERA_HEIGHT)
        )

        camera_bp.set_attribute(
            "fov",
            str(CAMERA_FOV)
        )

        camera_bp.set_attribute(
            "sensor_tick",
            str(FIXED_DELTA_SECONDS)
        )

        camera = world.spawn_actor(
            camera_bp,

            carla.Transform(

                carla.Location(
                    x=CAMERA_X,
                    y=CAMERA_Y,
                    z=CAMERA_Z
                ),

                carla.Rotation(
                    roll=CAMERA_ROLL,
                    pitch=CAMERA_PITCH,
                    yaw=CAMERA_YAW
                )
            ),

            attach_to=vehicle
        )

        camera.listen(
            camera_callback
        )

        # ====================================================
        # IMU
        # ====================================================

        imu_bp = (
            library.find(
                "sensor.other.imu"
            )
        )

        imu_bp.set_attribute(
            "sensor_tick",
            str(IMU_SENSOR_TICK)
        )

        imu = world.spawn_actor(

            imu_bp,

            carla.Transform(

                carla.Location(
                    x=0.0,
                    y=0.0,
                    z=0.0
                )
            ),

            attach_to=vehicle
        )

        imu.listen(
            imu_callback
        )

        print()
        print(
            "PDM-Lite sensors:"
        )

        print(
            f"  Camera: "
            f"{CAMERA_WIDTH}x"
            f"{CAMERA_HEIGHT} "
            f"FOV={CAMERA_FOV}"
        )

        print(
            f"  Camera position: "
            f"{CAMERA_X}, "
            f"{CAMERA_Y}, "
            f"{CAMERA_Z}"
        )

        print(
            "  Camera rotation: "
            "roll=0 pitch=0 yaw=0"
        )

        print(
            "  IMU: compass"
        )

        # ====================================================
        # PRIME
        # ====================================================

        print(
            "Waiting for synchronized sensors..."
        )

        frame = world.tick()

        camera_data = (
            get_sensor_frame_data(
                camera_queue,
                frame
            )
        )

        imu_data = (
            get_sensor_frame_data(
                imu_queue,
                frame
            )
        )

        (
            compass,
            initial_theta
        ) = (
            get_pdm_theta(
                imu_data
            )
        )

        vehicle_yaw = (
            vehicle
            .get_transform()
            .rotation
            .yaw
        )

        print(
            f"Vehicle yaw: "
            f"{vehicle_yaw:+.6f} deg"
        )

        print(
            f"IMU compass: "
            f"{compass:+.9f} rad"
        )

        print(
            f"Initial PDM-Lite theta: "
            f"{initial_theta:+.9f}"
        )

        print(
            "Camera ready."
        )

        # ====================================================
        # DIRECT ROUTE STATE
        # ====================================================

        route_state = CarlaRouteState(
            world,
            vehicle,
            global_route
        )

        # ====================================================
        # INITIAL BRAKE
        # ====================================================

        vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0
            )
        )

        world.tick()

        # ====================================================
        # DRIVING
        # ====================================================

        print()
        print("=" * 82)
        print("DRIVING")
        print("=" * 82)

        print(
            "Navigation = direct CARLA "
            "GlobalRoutePlanner"
        )

        print(
            "Angle = current vehicle yaw "
            "to route lookahead"
        )

        print(
            "Lateral = current vehicle position "
            "relative to route"
        )

        print(
            "Theta = IMU model feature"
        )

        print(
            "Camera = unchanged"
        )

        print(
            "Press Q to stop."
        )

        print(
            "Destination:",
            destination.location
        )

        print(
            "Route decisions:",
            route_turn_summary(
                global_route
            )
            or
            ["STRAIGHT"]
        )

        # ====================================================
        # MAIN LOOP
        # ====================================================

        while True:

            frame = world.tick()

            camera_data = (
                get_sensor_frame_data(
                    camera_queue,
                    frame
                )
            )

            imu_data = (
                get_sensor_frame_data(
                    imu_queue,
                    frame
                )
            )

            # ------------------------------------------------
            # IMAGE
            # ------------------------------------------------

            rgb = image_to_rgb(
                camera_data
            )

            # ------------------------------------------------
            # THETA
            # ------------------------------------------------

            (
                compass,
                theta
            ) = (
                get_pdm_theta(
                    imu_data
                )
            )

            # ------------------------------------------------
            # ROUTE
            # ------------------------------------------------

            route_state.update()

            # ------------------------------------------------
            # TELEMETRY
            # ------------------------------------------------

            (
                telemetry,
                raw
            ) = build_model_telemetry(
                vehicle,
                world,
                route_state,
                theta
            )

            # ------------------------------------------------
            # HARD SANITY CHECKS
            # ------------------------------------------------

            if not (
                -1.0
                <=
                float(telemetry[7])
                <=
                1.0
            ):

                raise RuntimeError(
                    "Angle outside model range: "
                    f"{telemetry[7]}"
                )

            if not (
                -1.0
                <=
                float(telemetry[9])
                <=
                1.0
            ):

                raise RuntimeError(
                    "Theta outside model range: "
                    f"{telemetry[9]}"
                )

            if not (
                -1.0
                <=
                float(telemetry[10])
                <=
                1.0
            ):

                raise RuntimeError(
                    "Lateral outside model range: "
                    f"{telemetry[10]}"
                )

            if not np.all(
                np.isfinite(
                    telemetry
                )
            ):

                raise RuntimeError(
                    "Telemetry contains NaN/Inf."
                )

            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            (
                steering,
                throttle,
                brake_logit,
                brake_probability
            ) = run_model(
                model,
                rgb,
                telemetry
            )

            # ------------------------------------------------
            # CONTROL
            # ------------------------------------------------

            (
                applied_steering,
                applied_throttle,
                applied_brake
            ) = apply_control(
                vehicle,
                steering,
                throttle,
                brake_probability
            )

            # ------------------------------------------------
            # DEBUG
            # ------------------------------------------------

            if PRINT_TELEMETRY:

                print_debug(
                    telemetry,
                    raw,
                    vehicle,
                    compass,
                    theta,
                    applied_steering,
                    applied_throttle,
                    brake_logit,
                    brake_probability
                )

            # ------------------------------------------------
            # CAMERA WINDOW
            # ------------------------------------------------

            show_camera(
                rgb,
                telemetry,
                raw,
                applied_steering,
                applied_throttle,
                brake_probability
            )

            if (
                SHOW_CAMERA
                and
                (
                    cv2.waitKey(1)
                    &
                    0xFF
                )
                ==
                ord("q")
            ):

                print(
                    "\nQ pressed."
                )

                break

            # ------------------------------------------------
            # DESTINATION
            # ------------------------------------------------

            if PRINT_TELEMETRY:

                distance_to_destination = (
                    vehicle
                    .get_location()
                    .distance(
                        destination.location
                    )
                )

                print(
                    f"Distance to destination: "
                    f"{distance_to_destination:.2f} m"
                )

    except KeyboardInterrupt:

        print(
            "\nKeyboard interrupt."
        )

    except Exception as exc:

        print()
        print("=" * 82)
        print("DRIVER ERROR")
        print("=" * 82)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        traceback.print_exc()

    finally:

        print()
        print("=" * 82)
        print("SHUTTING DOWN")
        print("=" * 82)

        # ----------------------------------------------------
        # BRAKE
        # ----------------------------------------------------

        if vehicle is not None:

            try:

                vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        steer=0.0,
                        brake=1.0
                    )
                )

            except Exception:
                pass

        # ----------------------------------------------------
        # SENSORS
        # ----------------------------------------------------

        for sensor in (
            camera,
            imu
        ):

            if sensor is not None:

                try:
                    sensor.stop()
                except Exception:
                    pass

                try:
                    sensor.destroy()
                except Exception:
                    pass

        # ----------------------------------------------------
        # VEHICLE
        # ----------------------------------------------------

        if vehicle is not None:

            try:
                vehicle.destroy()
            except Exception:
                pass

        # ----------------------------------------------------
        # WORLD SETTINGS
        # ----------------------------------------------------

        if (
            world is not None
            and
            original_settings is not None
        ):

            try:

                world.apply_settings(
                    original_settings
                )

            except Exception:
                pass

        # ----------------------------------------------------
        # WINDOWS
        # ----------------------------------------------------

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # ----------------------------------------------------
        # CWD
        # ----------------------------------------------------

        try:
            os.chdir(
                original_cwd
            )
        except Exception:
            pass

        print(
            "Driver stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()