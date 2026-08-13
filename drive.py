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
# PATHS / MODEL
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(PROJECT_DIR, "model_v8.pth")

DRIVELM_ROOT = os.path.expanduser("~/DriveLM-CARLA")
PDM_ROOT = os.path.join(DRIVELM_ROOT, "pdm_lite")
PDM_TEAM_CODE = os.path.join(PDM_ROOT, "team_code")

CARLA_ROOT = os.path.expanduser("~/carla-simulator")
CARLA_PYTHON_API = os.path.join(CARLA_ROOT, "PythonAPI", "carla")

for path in (PDM_TEAM_CODE, CARLA_PYTHON_API):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

try:
    from config import GlobalConfig
    from nav_planner import RoutePlanner
    from privileged_route_planner import PrivilegedRoutePlanner
    import transfuser_utils as t_u
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from agents.navigation.local_planner import RoadOption
except Exception as exc:
    raise RuntimeError(
        "\nCould not import official DriveLM/PDM-Lite modules.\n"
        f"PDM path: {PDM_TEAM_CODE}\n"
        f"Original error: {exc}"
    )

from model import AutonomousDriver

# ============================================================
# CONFIG
# ============================================================

CARLA_HOST = "localhost"
CARLA_PORT = 2000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_WIDTH = 400
MODEL_HEIGHT = 200
MODEL_TELEMETRY = 11

# ORIGINAL PDM-LITE CAMERA
CAMERA_X = -1.5
CAMERA_Y = 0.0
CAMERA_Z = 2.0
CAMERA_ROLL = 0.0
CAMERA_PITCH = 0.0
CAMERA_YAW = 0.0

CAMERA_WIDTH = 1024
CAMERA_HEIGHT = 512
CAMERA_FOV = 110.0

IMU_SENSOR_TICK = 0.05
FIXED_DELTA_SECONDS = 0.05

VEHICLE_FILTER = "vehicle.tesla.model3"

BRAKE_THRESHOLD = 0.50
MAX_THROTTLE = 0.40

# Steering actuator stabilization.
# The model output is still used directly in sign and magnitude,
# but rapid frame-to-frame jumps are limited.
STEER_SMOOTHING = 0.35
MAX_STEER_STEP = 0.12

SHOW_CAMERA = True
PRINT_TELEMETRY = True

TOWNS = ["Town01", "Town02", "Town03"]

MIN_ROUTE_DISTANCE = 180.0
MAX_ROUTE_DISTANCE = 1500.0
DESTINATION_ATTEMPTS = 150
REQUIRE_TURN = True

COMMAND_NAMES = {
    1: "LEFT",
    2: "RIGHT",
    3: "STRAIGHT",
    4: "LANEFOLLOW",
    5: "CHANGE_LANE_LEFT",
    6: "CHANGE_LANE_RIGHT",
}

IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((MODEL_HEIGHT, MODEL_WIDTH)),
    transforms.ToTensor(),
])

camera_queue = queue.Queue()
imu_queue = queue.Queue()

# Last applied steering command.  This is deliberately reset at startup.
_last_steering = 0.0


# ============================================================
# HELPERS
# ============================================================

def command_name(command):
    try:
        return COMMAND_NAMES.get(int(command), "UNKNOWN")
    except Exception:
        return "UNKNOWN"


def safe_float(value, name, fallback=None):
    if value is None:
        if fallback is not None:
            return float(fallback)
        raise RuntimeError(f"{name} is None.")

    try:
        result = float(value)
    except (TypeError, ValueError):
        if fallback is not None:
            return float(fallback)
        raise RuntimeError(f"{name} is not numeric: {value!r}")

    if not math.isfinite(result):
        if fallback is not None:
            return float(fallback)
        raise RuntimeError(f"{name} is not finite: {result}")

    return result


def camera_callback(image):
    camera_queue.put(image)


def imu_callback(measurement):
    imu_queue.put(measurement)


def get_sensor_frame_data(sensor_queue, frame):
    while True:
        try:
            data = sensor_queue.get(timeout=3.0)
        except queue.Empty:
            raise RuntimeError(
                f"Timed out waiting for sensor frame {frame}."
            )

        if data.frame < frame:
            continue

        if data.frame == frame:
            return data

        raise RuntimeError(
            f"Sensor frame mismatch: wanted {frame}, got {data.frame}."
        )


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

    # CARLA BGRA -> RGB
    return raw[:, :, :3][:, :, ::-1].copy()


def get_speed_ms(vehicle):
    return safe_float(
        vehicle.get_velocity().length(),
        "vehicle speed",
        0.0
    )


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


def get_pdm_theta(imu_measurement):
    compass = safe_float(
        imu_measurement.compass,
        "IMU compass"
    )

    theta = safe_float(
        t_u.preprocess_compass(compass),
        "processed PDM theta"
    )

    return compass, theta


# ============================================================
# WORLD / VEHICLE
# ============================================================

def load_navigation_town(client, town):

    if town not in TOWNS:
        raise RuntimeError(
            f"Invalid town {town}. Use one of {TOWNS}."
        )

    print()
    print("=" * 82)
    print("LOADING NAVIGATION TOWN")
    print("=" * 82)
    print("Selected town:", town)

    world = client.load_world(town)
    client.set_timeout(10.0)

    expected = f"Carla/Maps/{town}"
    start = time.time()

    while time.time() - start < 30.0:

        loaded = world.get_map().name

        if loaded == expected:
            print("Active map:", loaded)
            return world

        time.sleep(0.1)

    raise RuntimeError(
        f"Failed to load {town}.\n"
        f"Expected: {expected}\n"
        f"Current: {world.get_map().name}"
    )


def spawn_vehicle(world):

    blueprints = (
        world
        .get_blueprint_library()
        .filter(VEHICLE_FILTER)
    )

    if not blueprints:
        raise RuntimeError(
            f"Vehicle blueprint not found: {VEHICLE_FILTER}"
        )

    points = list(
        world
        .get_map()
        .get_spawn_points()
    )

    if not points:
        raise RuntimeError(
            "No CARLA spawn points."
        )

    random.shuffle(points)

    for transform in points:

        vehicle = world.try_spawn_actor(
            blueprints[0],
            transform
        )

        if vehicle is not None:

            vehicle.set_autopilot(False)

            print()
            print("Spawned vehicle:")
            print("  Type:", vehicle.type_id)
            print("  ID:", vehicle.id)
            print("  Location:", vehicle.get_location())
            print("  Rotation:", vehicle.get_transform().rotation)

            return vehicle

    raise RuntimeError(
        "Could not spawn ego vehicle."
    )


# ============================================================
# GLOBAL ROUTE
# ============================================================

def normalize_pdm_route(route):

    if route is None:
        return []

    normalized = []

    for index, item in enumerate(route):

        if (
            not isinstance(item, (tuple, list))
            or len(item) != 2
        ):
            raise RuntimeError(
                f"Invalid route item at index {index}."
            )

        position, road_option = item

        if isinstance(position, carla.Waypoint):
            transform = position.transform

        elif isinstance(position, carla.Transform):
            transform = position

        elif hasattr(position, "transform"):
            transform = position.transform

        else:
            raise RuntimeError(
                f"Invalid route position at index {index}: "
                f"{type(position).__name__}"
            )

        normalized.append(
            (transform, road_option)
        )

    return normalized


def build_global_route(
    world,
    start_transform,
    destination_transform
):

    planner = GlobalRoutePlanner(
        world.get_map(),
        2.0
    )

    raw_route = planner.trace_route(
        start_transform.location,
        destination_transform.location
    )

    return normalize_pdm_route(
        raw_route
    )


def route_length(route):

    if route is None or len(route) < 2:
        return 0.0

    total = 0.0

    for i in range(1, len(route)):

        total += (
            route[i - 1][0]
            .location
            .distance(
                route[i][0].location
            )
        )

    return total


def route_has_turn(route):

    return any(
        option in (
            RoadOption.LEFT,
            RoadOption.RIGHT
        )
        for _, option in route
    )


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


def choose_destination(world, vehicle):

    spawn_points = list(
        world
        .get_map()
        .get_spawn_points()
    )

    if len(spawn_points) < 2:
        raise RuntimeError(
            "Not enough CARLA spawn points."
        )

    start = vehicle.get_transform()

    candidates = list(spawn_points)
    random.shuffle(candidates)

    fallback = None

    print()
    print("=" * 82)
    print("SEARCHING FOR NAVIGATION DESTINATION")
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

        if straight_distance < MIN_ROUTE_DISTANCE:
            continue

        if straight_distance > MAX_ROUTE_DISTANCE:
            continue

        try:
            route = build_global_route(
                world,
                start,
                destination
            )
        except Exception:
            continue

        if len(route) < 10:
            continue

        length = route_length(route)

        if length < MIN_ROUTE_DISTANCE:
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
            or ["STRAIGHT"]
        )

        return destination, route

    if fallback is None:
        raise RuntimeError(
            "Could not find a valid destination route."
        )

    destination, route, length = fallback

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
        or ["STRAIGHT"]
    )

    return destination, route


# ============================================================
# OFFICIAL PDM ROUTE SYSTEM
# ============================================================

class PdmRouteSystem:

    def __init__(
        self,
        world,
        vehicle,
        global_plan
    ):

        self.world = world
        self.vehicle = vehicle

        self.config = GlobalConfig()

        self.command_planner = RoutePlanner(
            self.config.route_planner_min_distance,
            self.config.route_planner_max_distance
        )

        self.privileged_planner = (
            PrivilegedRoutePlanner(
                self.config
            )
        )

        self.global_plan = (
            normalize_pdm_route(
                global_plan
            )
        )

        if not self.global_plan:
            raise RuntimeError(
                "PDM-Lite received an empty global route."
            )

        self.command_planner.set_route(
            self.global_plan
        )

        first_location = (
            self.global_plan[0][0]
            .location
        )

        starts_with_parking_exit = (
            first_location.distance(
                vehicle.get_location()
            ) > 2.0
        )

        self.privileged_planner.setup_route(
            self.global_plan,
            world,
            world.get_map(),
            starts_with_parking_exit,
            vehicle.get_location()
        )

        self.privileged_planner.save()

        self.command = None
        self.next_command = None

        self.target_point = None
        self.next_target_point = None

        self.route_np = None
        self.route_wp = None

        self.aim_wp = None

        self.speed_limit = None

        self.distance_to_light = None
        self.next_light = None

        self.distance_to_stop = None
        self.next_stop = None

    # --------------------------------------------------------
    # SPEED LIMIT
    # --------------------------------------------------------

    def get_speed_limit(self):

        planner = self.privileged_planner

        route_index = getattr(
            planner,
            "route_index",
            None
        )

        speed_limits = getattr(
            planner,
            "speed_limits",
            None
        )

        if (
            route_index is not None
            and
            speed_limits is not None
            and
            len(speed_limits) > 0
        ):

            try:

                index = int(
                    route_index
                )

                index = max(
                    0,
                    min(
                        index,
                        len(speed_limits) - 1
                    )
                )

                value = speed_limits[index]

                if value is not None:

                    value = float(value)

                    if (
                        math.isfinite(value)
                        and
                        value > 0.0
                    ):
                        return value

            except (
                TypeError,
                ValueError,
                IndexError
            ):
                pass

        # IMPORTANT:
        # Never call float(None).
        return get_carla_speed_limit_ms(
            self.vehicle
        )

    # --------------------------------------------------------
    # REMAINING ROUTE
    # --------------------------------------------------------

    def get_remaining_route(self):

        points = getattr(
            self.privileged_planner,
            "route_points",
            None
        )

        if points is None:
            return None

        points = np.asarray(
            points,
            dtype=np.float64
        )

        if (
            points.ndim != 2
            or
            points.shape[0] == 0
        ):
            return None

        try:
            index = int(
                getattr(
                    self.privileged_planner,
                    "route_index",
                    0
                )
            )

        except Exception:
            index = 0

        index = max(
            0,
            min(
                index,
                points.shape[0] - 1
            )
        )

        return points[index:]

    # --------------------------------------------------------
    # LOOKAHEAD
    # --------------------------------------------------------

    def get_aim_point(self):

        route = self.route_np

        if route is None:
            return None

        route = np.asarray(
            route,
            dtype=np.float64
        )

        if (
            route.ndim != 2
            or
            route.shape[0] == 0
        ):
            return None

        speed_kmh = (
            get_speed_ms(
                self.vehicle
            ) * 3.6
        )

        lookahead = (
            float(
                self.config
                .lateral_pid_speed_scale
            )
            * speed_kmh
            +
            float(
                self.config
                .lateral_pid_speed_offset
            )
        )

        lookahead = float(
            np.clip(
                lookahead,
                float(
                    self.config
                    .lateral_pid_default_lookahead
                ),
                float(
                    self.config
                    .lateral_pid_maximum_lookahead_distance
                )
            )
        )

        index = int(
            min(
                lookahead,
                route.shape[0] - 1
            )
        )

        return route[index].copy()

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(self):

        location = (
            self.vehicle
            .get_location()
        )

        ego_position = np.array(
            [
                location.x,
                location.y,
                location.z
            ],
            dtype=np.float64
        )

        result = (
            self.privileged_planner
            .run_step(
                ego_position
            )
        )

        if len(result) < 8:
            raise RuntimeError(
                "Unexpected PDM-Lite "
                "PrivilegedRoutePlanner result "
                f"length: {len(result)}"
            )

        self.route_np = result[0]
        self.route_wp = result[1]

        self.distance_to_light = result[3]
        self.next_light = result[4]

        self.distance_to_stop = result[5]
        self.next_stop = result[6]

        self.speed_limit = (
            self.get_speed_limit()
        )

        if self.route_np is None:
            self.route_np = (
                self.get_remaining_route()
            )

        if self.route_np is None:
            raise RuntimeError(
                "Official PDM-Lite returned "
                "no route points."
            )

        self.route_np = np.asarray(
            self.route_np,
            dtype=np.float64
        )

        if (
            self.route_np.ndim != 2
            or
            self.route_np.shape[0] == 0
        ):
            self.route_np = (
                self.get_remaining_route()
            )

        if (
            self.route_np is None
            or
            self.route_np.shape[0] == 0
        ):
            raise RuntimeError(
                "PDM-Lite has no remaining "
                "route points."
            )

        # ----------------------------------------------------
        # COMMAND
        # ----------------------------------------------------

        command_route = (
            self.command_planner
            .run_step(
                ego_position
            )
        )

        if not command_route:
            raise RuntimeError(
                "Official RoutePlanner "
                "returned no route."
            )

        if len(command_route) >= 3:

            (
                target_point,
                command_option
            ) = command_route[1]

            (
                next_target_point,
                next_command_option
            ) = command_route[2]

        elif len(command_route) == 2:

            (
                target_point,
                command_option
            ) = command_route[1]

            next_target_point = target_point
            next_command_option = command_option

        else:

            (
                target_point,
                command_option
            ) = command_route[0]

            next_target_point = target_point
            next_command_option = command_option

        self.target_point = np.asarray(
            target_point,
            dtype=np.float64
        )

        self.next_target_point = np.asarray(
            next_target_point,
            dtype=np.float64
        )

        self.command = int(
            command_option.value
        )

        self.next_command = int(
            next_command_option.value
        )

        self.aim_wp = (
            self.get_aim_point()
        )

        if self.aim_wp is None:
            raise RuntimeError(
                "Could not determine "
                "PDM-Lite lateral aim waypoint."
            )

    # --------------------------------------------------------
    # ANGLE
    # --------------------------------------------------------

    def calculate_angle(self, theta):

        if self.aim_wp is None:
            self.aim_wp = (
                self.get_aim_point()
            )

        if self.aim_wp is None:
            raise RuntimeError(
                "No PDM-Lite aim waypoint."
            )

        location = (
            self.vehicle
            .get_location()
        )

        current = np.array(
            [
                location.x,
                location.y
            ],
            dtype=np.float64
        )

        target = self.aim_wp[:2]

        delta = target - current

        cos_heading = math.cos(theta)
        sin_heading = math.sin(theta)

        # Global -> ego-relative
        aim_x = (
            cos_heading * delta[0]
            +
            sin_heading * delta[1]
        )

        aim_y = (
            -sin_heading * delta[0]
            +
            cos_heading * delta[1]
        )

        # IMPORTANT: the training dataset stores `angle` in RADIANS.
        # Do not convert this to degrees or divide by 90.
        # The previous live driver returned degrees/90 here, which
        # changed the feature units and could make the network produce
        # a large, incorrect steering command.
        angle = math.atan2(aim_y, aim_x)

        return float(
            np.clip(angle, -math.pi, math.pi)
        )

    # --------------------------------------------------------
    # V8 LATERAL DISTANCE
    # --------------------------------------------------------

    def calculate_lateral_distance(self):

        """
        V8 live equivalent of the dataset lateral feature.

        Dataset route coordinates are ego-local.
        CARLA PDM route coordinates are global.

        We therefore find the closest PDM route segment in
        world coordinates and compute the signed distance from
        the ego vehicle to that segment.

        negative = left
        positive = right
        """

        route = self.route_np

        if route is None:
            return 0.0

        route = np.asarray(
            route,
            dtype=np.float64
        )

        if (
            route.ndim != 2
            or
            route.shape[0] < 2
            or
            route.shape[1] < 2
        ):
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

        points = route[:, :2]

        best_dist2 = float(
            "inf"
        )

        best_lateral = 0.0

        for i in range(
            len(points) - 1
        ):

            p1 = points[i]
            p2 = points[i + 1]

            segment = p2 - p1

            segment_length_sq = float(
                np.dot(
                    segment,
                    segment
                )
            )

            if segment_length_sq < 1e-9:
                continue

            t = float(
                np.dot(
                    ego - p1,
                    segment
                )
                /
                segment_length_sq
            )

            t = max(
                0.0,
                min(
                    1.0,
                    t
                )
            )

            closest = (
                p1
                +
                t * segment
            )

            delta = ego - closest

            distance_sq = float(
                np.dot(
                    delta,
                    delta
                )
            )

            length = math.sqrt(
                segment_length_sq
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

            # CARLA +Y is right when yaw=0.
            # Right normal to the route tangent.
            right_x = -ty
            right_y = tx

            lateral = float(
                delta[0] * right_x
                +
                delta[1] * right_y
            )

            if distance_sq < best_dist2:

                best_dist2 = distance_sq
                best_lateral = lateral

        return best_lateral


# ============================================================
# V8 TELEMETRY
# ============================================================

def get_speed_reduction_object():

    # Same semantic slots as training.
    # No artificial live object detector is invented.
    return (
        0,
        -1,
        0.0
    )


def build_model_telemetry(
    vehicle,
    world,
    route_system,
    theta
):

    speed_raw = (
        get_speed_ms(
            vehicle
        )
    )

    speed_limit_raw = safe_float(
        route_system.speed_limit,
        "speed limit",
        fallback=(
            get_carla_speed_limit_ms(
                vehicle
            )
        )
    )

    angle_raw = safe_float(
        route_system.calculate_angle(
            theta
        ),
        "PDM angle"
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

    lateral_raw = (
        route_system
        .calculate_lateral_distance()
    )

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
                route_system.command
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
                route_system.next_command
            ),

            # 9 theta
            np.clip(
                theta / math.pi,
                -1.0,
                1.0
            ),

            # 10 lateral distance
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
            f"Telemetry contains NaN/Inf:\n"
            f"{telemetry}"
        )

    raw = {
        "speed": speed_raw,
        "speed_limit": speed_limit_raw,
        "angle": angle_raw,
        "theta": theta,
        "lateral": lateral_raw,
        "aim_wp": (
            None
            if route_system.aim_wp is None
            else route_system.aim_wp.copy()
        ),
        "target_point": (
            None
            if route_system.target_point is None
            else route_system.target_point.copy()
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
            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:
            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        elif "model" in checkpoint:
            state_dict = (
                checkpoint[
                    "model"
                ]
            )

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

    return model


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

    if output.shape != (
        1,
        3
    ):
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
        torch.sigmoid(
            output[0, 2]
        ).item()
    )

    return (
        steering,
        throttle,
        brake_logit,
        brake_probability
    )


def apply_control(
    vehicle,
    steering,
    throttle,
    brake_probability
):
    global _last_steering

    # --------------------------------------------------------
    # Model output
    # --------------------------------------------------------
    steering = float(np.clip(steering, -1.0, 1.0))
    throttle = float(np.clip(throttle, 0.0, MAX_THROTTLE))

    # --------------------------------------------------------
    # Steering stabilization
    #
    # IMPORTANT:
    # We do NOT invert the model steering sign.
    # Training target is the CARLA steer value directly.
    #
    # We only prevent unrealistic one-frame jumps.
    # --------------------------------------------------------
    delta = steering - _last_steering

    delta = float(
        np.clip(
            delta,
            -MAX_STEER_STEP,
            MAX_STEER_STEP
        )
    )

    limited = _last_steering + delta

    steering_applied = (
        STEER_SMOOTHING * limited
        +
        (1.0 - STEER_SMOOTHING) * _last_steering
    )

    steering_applied = float(
        np.clip(
            steering_applied,
            -1.0,
            1.0
        )
    )

    _last_steering = steering_applied

    # --------------------------------------------------------
    # Brake
    # --------------------------------------------------------
    brake = (
        brake_probability >= BRAKE_THRESHOLD
    )

    if brake:
        throttle = 0.0

    control = carla.VehicleControl(
        steer=steering_applied,
        throttle=throttle,
        brake=1.0 if brake else 0.0,
        hand_brake=False,
        reverse=False,
        manual_gear_shift=False
    )

    vehicle.apply_control(control)

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

        f"Angle: {raw['angle']:+.3f}",
        f"Theta: {raw['theta']:+.3f}",
        f"Lateral: {raw['lateral']:+.3f} m",
        f"Steer: {steering:+.3f}",
        f"Throttle: {throttle:.3f}",
        f"Brake P: {brake_probability:.3f}",
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

    # Steering indicator.
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

    yaw = (
        vehicle
        .get_transform()
        .rotation
        .yaw
    )

    print(
        "\033[2J\033[H",
        end=""
    )

    print(
        "=" * 82
    )

    print(
        "                 PDM-LITE NEURAL DRIVER V8"
    )

    print(
        "=" * 82
    )

    print(
        f"Vehicle yaw    : "
        f"{yaw:+.6f} deg"
    )

    print(
        f"IMU compass    : "
        f"{compass:+.9f} rad"
    )

    print(
        f"PDM theta raw  : "
        f"{theta:+.9f} rad"
    )

    print(
        "-" * 82
    )

    print(
        "RAW PDM VALUES"
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
        f"{raw['angle']:+.6f}"
    )

    print(
        f"Lateral raw     : "
        f"{raw['lateral']:+.6f} m"
    )

    print(
        f"Aim WP          : "
        f"{raw['aim_wp']}"
    )

    print(
        "-" * 82
    )

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

    for i, label in enumerate(
        labels
    ):

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

    print(
        "-" * 82
    )

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

    print(
        "=" * 82
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "PDM-Lite Neural Driver V8 FIXED"
        )
    )

    parser.add_argument(
        "--town",
        choices=TOWNS,
        default=None,
        help=(
            "CARLA town. If omitted, "
            "Town01/02/03 is selected randomly."
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

        print()
        print(
            "=" * 82
        )

        print(
            "                 PDM-LITE NEURAL DRIVER V8"
        )

        print(
            "=" * 82
        )

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
            "Camera     : 1024 x 512 / FOV 110"
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
            "Lateral    : route-based normalized"
        )

        print(
            "Theta      : "
            "CARLA IMU -> preprocess_compass()"
        )

        print(
            "=" * 82
        )

        # ----------------------------------------------------
        # CARLA
        # ----------------------------------------------------

        client = carla.Client(
            CARLA_HOST,
            CARLA_PORT
        )

        client.set_timeout(
            10.0
        )

        world = load_navigation_town(
            client,
            selected_town
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

        # ----------------------------------------------------
        # SYNCHRONOUS MODE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # PDM WORKING DIRECTORY
        # ----------------------------------------------------

        os.chdir(
            PDM_ROOT
        )

        print(
            "PDM-Lite working directory:",
            os.getcwd()
        )

        # ----------------------------------------------------
        # VEHICLE
        # ----------------------------------------------------

        vehicle = spawn_vehicle(
            world
        )

        # ----------------------------------------------------
        # ROUTE
        # ----------------------------------------------------

        destination, global_route = (
            choose_destination(
                world,
                vehicle
            )
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

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = load_model()

        library = (
            world
            .get_blueprint_library()
        )

        # ====================================================
        # CAMERA
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

        # ----------------------------------------------------
        # PRIME SENSORS
        # ----------------------------------------------------

        print(
            "Waiting for synchronized sensors..."
        )

        frame = (
            world.tick()
        )

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

        compass, initial_theta = (
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

        # ----------------------------------------------------
        # OFFICIAL PDM ROUTE SYSTEM
        # ----------------------------------------------------

        print()
        print(
            "Initializing official "
            "PDM-Lite route planners..."
        )

        route_system = (
            PdmRouteSystem(
                world,
                vehicle,
                global_route
            )
        )

        # ----------------------------------------------------
        # INITIAL STOP
        # ----------------------------------------------------

        vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0
            )
        )

        world.tick()

        # ----------------------------------------------------
        # DRIVE
        # ----------------------------------------------------

        print()
        print(
            "=" * 82
        )

        print(
            "DRIVING"
        )

        print(
            "=" * 82
        )

        print(
            "Town:",
            selected_town
        )

        print(
            "Command = official RoutePlanner."
        )

        print(
            "Angle = official PDM-Lite "
            "route lookahead."
        )

        print(
            "Theta = CARLA IMU -> "
            "official preprocess_compass()."
        )

        print(
            "Lateral = V8 route-based distance."
        )

        print(
            "Telemetry = 11 values."
        )

        print(
            "Brake is predicted by the model."
        )

        print(
            "Press Q to stop. Ctrl+C also stops the driver."
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

        # ----------------------------------------------------
        # MAIN LOOP
        # ----------------------------------------------------

        while True:

            frame = (
                world.tick()
            )

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
            # CAMERA
            # ------------------------------------------------

            rgb = image_to_rgb(
                camera_data
            )

            # ------------------------------------------------
            # IMU / THETA
            # ------------------------------------------------

            compass, theta = (
                get_pdm_theta(
                    imu_data
                )
            )

            # ------------------------------------------------
            # ROUTE
            # ------------------------------------------------

            route_system.update()

            # ------------------------------------------------
            # V8 TELEMETRY
            # ------------------------------------------------

            (
                telemetry,
                raw
            ) = build_model_telemetry(
                vehicle,
                world,
                route_system,
                theta
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
                    & 0xFF
                )
                == ord("q")
            ):

                print(
                    "\nQ pressed."
                )

                break

            # ------------------------------------------------
            # DESTINATION
            # ------------------------------------------------
            #
            # Do NOT terminate automatically near the destination.
            # Keep the neural driver running for as long as possible.
            # Press Q or Ctrl+C to stop.
            distance_to_destination = (
                vehicle
                .get_location()
                .distance(
                    destination.location
                )
            )

            if PRINT_TELEMETRY:
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
        print(
            "=" * 82
        )

        print(
            "DRIVER ERROR"
        )

        print(
            "=" * 82
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        traceback.print_exc()

    finally:

        print()
        print(
            "=" * 82
        )

        print(
            "SHUTTING DOWN"
        )

        print(
            "=" * 82
        )

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
        # RESTORE SETTINGS
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


if __name__ == "__main__":
    main()