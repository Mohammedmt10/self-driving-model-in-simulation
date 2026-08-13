import os
import sys
import math
import time
import queue
import random
import traceback

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
    "model_v5.pth"
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


# ============================================================
# PYTHON PATH
# ============================================================

for path in (
    PDM_TEAM_CODE,
    CARLA_PYTHON_API,
):

    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


# ============================================================
# OFFICIAL PDM-LITE IMPORTS
# ============================================================

try:

    from config import GlobalConfig

    from nav_planner import RoutePlanner

    from privileged_route_planner import (
        PrivilegedRoutePlanner
    )

    import transfuser_utils as t_u

except Exception as exc:

    raise RuntimeError(
        "\n"
        "Could not import official DriveLM/PDM-Lite modules.\n\n"
        f"PDM path:\n{PDM_TEAM_CODE}\n\n"
        f"Original error:\n{exc}"
    )


# ============================================================
# MODEL
# ============================================================

from model import AutonomousDriver


# ============================================================
# CARLA
# ============================================================

CARLA_HOST = "localhost"
CARLA_PORT = 2000


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL INPUT
# ============================================================

MODEL_WIDTH = 400
MODEL_HEIGHT = 200

MODEL_TELEMETRY = 10


# ============================================================
# PDM-LITE CAMERA
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
# PDM-LITE IMU
# ============================================================

IMU_X = 0.0
IMU_Y = 0.0
IMU_Z = 0.0

IMU_SENSOR_TICK = 0.05


# ============================================================
# SYNCHRONOUS SIMULATION
# ============================================================

FIXED_DELTA_SECONDS = 0.05


# ============================================================
# VEHICLE
# ============================================================

VEHICLE_FILTER = "vehicle.tesla.model3"


# ============================================================
# CONTROL
# ============================================================

BRAKE_THRESHOLD = 0.50

MAX_THROTTLE = 0.4


# ============================================================
# DISPLAY
# ============================================================

SHOW_CAMERA = True
PRINT_TELEMETRY = True


# ============================================================
# COMMANDS
# ============================================================

COMMAND_NAMES = {
    1: "LEFT",
    2: "RIGHT",
    3: "STRAIGHT",
    4: "LANEFOLLOW",
    5: "CHANGE_LANE_LEFT",
    6: "CHANGE_LANE_RIGHT",
}


def command_name(command):

    try:

        command = int(command)

    except Exception:

        return "UNKNOWN"

    return COMMAND_NAMES.get(
        command,
        "UNKNOWN"
    )


# ============================================================
# IMAGE TRANSFORM
#
# EXACTLY MATCHES DATASET
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
# SENSOR CALLBACKS
# ============================================================

def camera_callback(image):

    camera_queue.put(
        image
    )


def imu_callback(measurement):

    imu_queue.put(
        measurement
    )


# ============================================================
# SENSOR SYNCHRONIZATION
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
                f"Timed out waiting for "
                f"sensor frame {frame}."
            )

        if data.frame < frame:

            continue

        if data.frame == frame:

            return data

        raise RuntimeError(
            f"Sensor frame mismatch: "
            f"wanted {frame}, got {data.frame}."
        )


# ============================================================
# SAFE NUMERIC
# ============================================================

def safe_float(
    value,
    name,
    fallback=None
):

    if value is None:

        if fallback is not None:

            return float(
                fallback
            )

        raise RuntimeError(
            f"{name} is None."
        )

    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        if fallback is not None:

            return float(
                fallback
            )

        raise RuntimeError(
            f"{name} is not numeric: "
            f"{value!r}"
        )

    if not math.isfinite(
        result
    ):

        if fallback is not None:

            return float(
                fallback
            )

        raise RuntimeError(
            f"{name} is not finite: "
            f"{result}"
        )

    return result


# ============================================================
# IMAGE CONVERSION
# ============================================================

def image_to_rgb(
    image
):

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
    return (
        raw[:, :, :3][:, :, ::-1]
        .copy()
    )


# ============================================================
# SPEED
# ============================================================

def get_speed_ms(
    vehicle
):

    velocity = (
        vehicle.get_velocity()
    )

    return safe_float(
        velocity.length(),
        "vehicle speed"
    )


# ============================================================
# CARLA SPEED LIMIT
#
# CARLA API returns km/h.
# Convert to m/s.
# ============================================================

def get_carla_speed_limit_ms(
    vehicle
):

    value = (
        vehicle.get_speed_limit()
    )

    if value is None:

        return (
            50.0 / 3.6
        )

    value = float(
        value
    )

    if not math.isfinite(
        value
    ):

        return (
            50.0 / 3.6
        )

    if value <= 0.0:

        return (
            50.0 / 3.6
        )

    return (
        value / 3.6
    )


# ============================================================
# PDM-LITE THETA
#
# Uses CARLA IMU compass and the official
# DriveLM/PDM-Lite compass preprocessing.
# ============================================================

def get_pdm_theta(
    imu_measurement
):

    compass = safe_float(
        imu_measurement.compass,
        "IMU compass"
    )

    theta = (
        t_u.preprocess_compass(
            compass
        )
    )

    theta = safe_float(
        theta,
        "processed PDM theta"
    )

    return (
        compass,
        theta
    )


# ============================================================
# MAP HELPERS
# ============================================================

def short_map_name(
    map_name
):

    return map_name.split(
        "/"
    )[-1]


def ensure_supported_map(
    client
):

    world = client.get_world()

    current_name = (
        world.get_map().name
    )

    print(
        "Current map:",
        current_name
    )

    current_short = (
        short_map_name(
            current_name
        )
    )

    speed_limit_dir = os.path.join(
        PDM_TEAM_CODE,
        "speed_limits"
    )

    exact_asset = os.path.join(
        speed_limit_dir,
        f"{current_short}_speed_limits.npy"
    )

    # --------------------------------------------------------
    # Directly supported map
    # --------------------------------------------------------

    if (
        not current_short.endswith("_Opt")
        and os.path.isfile(
            exact_asset
        )
    ):

        print(
            "PDM-Lite speed-limit asset:",
            os.path.basename(
                exact_asset
            )
        )

        return world

    # --------------------------------------------------------
    # Town10HD_Opt -> official Town10HD asset
    # --------------------------------------------------------

    if current_short == "Town10HD_Opt":

        target_asset = os.path.join(
            speed_limit_dir,
            "Town10HD_speed_limits.npy"
        )

        if not os.path.isfile(
            target_asset
        ):

            raise RuntimeError(
                "Official Town10HD "
                "speed-limit asset is missing:\n"
                f"{target_asset}"
            )

        print()
        print(
            "Town10HD_Opt has no official "
            "PDM-Lite speed-limit asset."
        )

        print(
            "Loading official supported map: "
            "Town10HD"
        )

        world = client.load_world(
            "Town10HD"
        )

        client.set_timeout(
            10.0
        )

        expected = (
            "Carla/Maps/Town10HD"
        )

        start = time.time()

        while True:

            loaded = (
                world
                .get_map()
                .name
            )

            if loaded == expected:

                break

            if (
                time.time() - start
                > 30.0
            ):

                raise RuntimeError(
                    "Failed to load Town10HD.\n"
                    f"Expected: {expected}\n"
                    f"Current: {loaded}"
                )

            time.sleep(
                0.1
            )

        print(
            "Active map:",
            loaded
        )

        return world

    raise RuntimeError(
        "Unsupported CARLA map:\n"
        f"{current_name}"
    )


# ============================================================
# VEHICLE
# ============================================================

def spawn_vehicle(
    world
):

    blueprints = (
        world
        .get_blueprint_library()
        .filter(
            VEHICLE_FILTER
        )
    )

    if not blueprints:

        raise RuntimeError(
            f"Vehicle blueprint not found: "
            f"{VEHICLE_FILTER}"
        )

    blueprint = blueprints[0]

    spawn_points = (
        world
        .get_map()
        .get_spawn_points()
    )

    if not spawn_points:

        raise RuntimeError(
            "No CARLA spawn points."
        )

    points = list(
        spawn_points
    )

    random.shuffle(
        points
    )

    for transform in points:

        vehicle = (
            world.try_spawn_actor(
                blueprint,
                transform
            )
        )

        if vehicle is not None:

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
                vehicle.get_transform().rotation
            )

            return vehicle

    raise RuntimeError(
        "Could not spawn vehicle."
    )


# ============================================================
# FORWARD DESTINATION
# ============================================================

def build_forward_destination(
    world,
    vehicle,
    distance_m=250.0
):

    carla_map = (
        world.get_map()
    )

    location = (
        vehicle.get_location()
    )

    current_wp = (
        carla_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
    )

    if current_wp is None:

        raise RuntimeError(
            "No driving waypoint "
            "under vehicle."
        )

    steps = int(
        distance_m / 2.0
    )

    for _ in range(
        steps
    ):

        next_wps = (
            current_wp.next(
                2.0
            )
        )

        if not next_wps:

            break

        same_lane = [

            wp
            for wp in next_wps

            if (
                wp.road_id
                == current_wp.road_id
                and
                wp.lane_id
                == current_wp.lane_id
            )
        ]

        if same_lane:

            current_wp = (
                same_lane[0]
            )

        else:

            current_wp = (
                next_wps[0]
            )

    return (
        current_wp
        .transform
        .location
    )


# ============================================================
# GLOBAL ROUTE
# ============================================================

def build_global_plan(
    world,
    vehicle
):

    try:

        from agents.navigation.global_route_planner import (
            GlobalRoutePlanner
        )

    except Exception as exc:

        raise RuntimeError(
            "CARLA agents GlobalRoutePlanner "
            "could not be imported.\n"
            f"{exc}"
        )

    planner = (
        GlobalRoutePlanner(
            world.get_map(),
            2.0
        )
    )

    start = (
        vehicle.get_location()
    )

    destination = (
        build_forward_destination(
            world,
            vehicle
        )
    )

    print()
    print(
        "Forward route destination:"
    )

    print(
        f"  x={destination.x:.3f}"
    )

    print(
        f"  y={destination.y:.3f}"
    )

    print(
        f"  z={destination.z:.3f}"
    )

    route = (
        planner.trace_route(
            start,
            destination
        )
    )

    if not route:

        raise RuntimeError(
            "GlobalRoutePlanner returned "
            "an empty route."
        )

    plan = [

        (
            waypoint.transform,
            road_option
        )

        for waypoint, road_option
        in route
    ]

    print(
        "Global route points:",
        len(plan)
    )

    return plan


# ============================================================
# PDM ROUTE SYSTEM
# ============================================================

class PdmRouteSystem:

    def __init__(
        self,
        world,
        vehicle
    ):

        self.world = world
        self.vehicle = vehicle

        self.config = (
            GlobalConfig()
        )

        self.command_planner = (
            RoutePlanner(
                self.config.route_planner_min_distance,
                self.config.route_planner_max_distance
            )
        )

        self.privileged_planner = (
            PrivilegedRoutePlanner(
                self.config
            )
        )

        # ----------------------------------------------------
        # Build global route
        # ----------------------------------------------------

        self.global_plan = (
            build_global_plan(
                world,
                vehicle
            )
        )

        # ----------------------------------------------------
        # Official route planner
        # ----------------------------------------------------

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
            )
            > 2.0
        )

        # ----------------------------------------------------
        # Official privileged planner
        # ----------------------------------------------------

        self.privileged_planner.setup_route(
            self.global_plan,
            world,
            world.get_map(),
            starts_with_parking_exit,
            vehicle.get_location()
        )

        self.privileged_planner.save()

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

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

    # ========================================================
    # SPEED LIMIT
    # ========================================================

    def get_speed_limit(
        self
    ):

        planner = (
            self.privileged_planner
        )

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

                value = (
                    speed_limits[index]
                )

                if value is not None:

                    value = float(
                        value
                    )

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

        return (
            get_carla_speed_limit_ms(
                self.vehicle
            )
        )

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self
    ):

        location = (
            self.vehicle.get_location()
        )

        ego_position = np.array(
            [
                location.x,
                location.y,
                location.z
            ],
            dtype=np.float64
        )

        # ----------------------------------------------------
        # Official privileged planner
        # ----------------------------------------------------

        result = (
            self.privileged_planner.run_step(
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

        # ----------------------------------------------------
        # Route validation
        # ----------------------------------------------------

        if self.route_np is None:

            raise RuntimeError(
                "Official PDM-Lite returned "
                "route_np=None."
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
                self.get_remaining_route_from_planner()
            )

        if self.route_np is None:

            raise RuntimeError(
                "Official PDM-Lite has no "
                "remaining route points."
            )

        # ----------------------------------------------------
        # Official command planner
        # ----------------------------------------------------

        command_route = (
            self.command_planner.run_step(
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

            next_target_point = (
                target_point
            )

            next_command_option = (
                command_option
            )

        else:

            (
                target_point,
                command_option
            ) = command_route[0]

            next_target_point = (
                target_point
            )

            next_command_option = (
                command_option
            )

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

        # ----------------------------------------------------
        # Official-style PDM lookahead
        # ----------------------------------------------------

        self.aim_wp = (
            self.get_aim_point()
        )

        if self.aim_wp is None:

            raise RuntimeError(
                "Could not determine "
                "PDM-Lite lateral aim waypoint."
            )

    # ========================================================
    # REMAINING ROUTE FALLBACK
    # ========================================================

    def get_remaining_route_from_planner(
        self
    ):

        planner = (
            self.privileged_planner
        )

        points = getattr(
            planner,
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

        index = getattr(
            planner,
            "route_index",
            0
        )

        try:

            index = int(
                index
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

        remaining = (
            points[index:]
        )

        if remaining.shape[0] == 0:

            return None

        return remaining

    # ========================================================
    # LOOKAHEAD
    # ========================================================

    def get_aim_point(
        self
    ):

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

        speed_ms = (
            get_speed_ms(
                self.vehicle
            )
        )

        speed_kmh = (
            speed_ms * 3.6
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

        minimum = float(
            self.config
            .lateral_pid_default_lookahead
        )

        maximum = float(
            self.config
            .lateral_pid_maximum_lookahead_distance
        )

        lookahead = float(
            np.clip(
                lookahead,
                minimum,
                maximum
            )
        )

        index = int(
            min(
                lookahead,
                route.shape[0] - 1
            )
        )

        return np.asarray(
            route[index],
            dtype=np.float64
        )

    # ========================================================
    # ANGLE
    # ========================================================

    def calculate_angle(
        self,
        theta
    ):

        if self.aim_wp is None:

            self.aim_wp = (
                self.get_aim_point()
            )

        if self.aim_wp is None:

            raise RuntimeError(
                "No PDM-Lite aim waypoint."
            )

        location = (
            self.vehicle.get_location()
        )

        current = np.array(
            [
                location.x,
                location.y
            ],
            dtype=np.float64
        )

        target = (
            self.aim_wp[:2]
        )

        delta = (
            target - current
        )

        cos_heading = math.cos(
            theta
        )

        sin_heading = math.sin(
            theta
        )

        # Global -> ego-relative frame
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

        angle = (
            -math.atan2(
                -aim_y,
                aim_x
            )
        )

        return float(
            math.degrees(
                angle
            ) / 90.0
        )


# ============================================================
# OBJECT FEATURES
#
# IMPORTANT:
# The V5 model no longer receives brake as an input.
#
# These three object values retain the same semantic slots
# used by the training interface. This function deliberately
# does not invent a new object-detection system.
# ============================================================

def get_speed_reduction_object():

    return (
        0,
        -1,
        0.0
    )


# ============================================================
# BUILD MODEL TELEMETRY
#
# EXACTLY 10 VALUES FOR MODEL V5
#
# 0  speed
# 1  command
# 2  junction
# 3  object_type
# 4  object_id
# 5  object_distance
# 6  speed_limit
# 7  angle
# 8  next_command
# 9  theta
# ============================================================

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
        fallback=get_carla_speed_limit_ms(
            vehicle
        )
    )

    angle_raw = safe_float(
        route_system.calculate_angle(
            theta
        ),
        "PDM angle"
    )

    # --------------------------------------------------------
    # Junction
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Object features
    # --------------------------------------------------------

    (
        object_type,
        object_id,
        object_distance_raw
    ) = get_speed_reduction_object()

    # --------------------------------------------------------
    # SAME NORMALIZATION AS TRAINING
    # --------------------------------------------------------

    speed = np.clip(
        speed_raw / 30.0,
        0.0,
        2.0
    )

    speed_limit = np.clip(
        speed_limit_raw / 30.0,
        0.0,
        2.0
    )

    object_distance = np.clip(
        object_distance_raw / 50.0,
        0.0,
        1.0
    )

    angle = np.clip(
        angle_raw / math.pi,
        -1.0,
        1.0
    )

    theta_input = np.clip(
        theta / math.pi,
        -1.0,
        1.0
    )

    # --------------------------------------------------------
    # EXACT V5 TELEMETRY
    # --------------------------------------------------------

    telemetry = np.array(
        [
            speed,                      # 0
            float(route_system.command),# 1
            junction,                   # 2
            float(object_type),         # 3
            float(object_id),           # 4
            object_distance,            # 5
            speed_limit,                # 6
            angle,                      # 7
            float(route_system.next_command),  # 8
            theta_input                 # 9
        ],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # HARD VALIDATION
    # --------------------------------------------------------

    if telemetry.shape != (
        MODEL_TELEMETRY,
    ):

        raise RuntimeError(
            "Telemetry shape mismatch: "
            f"{telemetry.shape}; "
            f"expected ({MODEL_TELEMETRY},)"
        )

    if not np.all(
        np.isfinite(
            telemetry
        )
    ):

        raise RuntimeError(
            "Telemetry contains NaN/Inf:\n"
            f"{telemetry}"
        )

    raw = {

        "speed": speed_raw,

        "speed_limit": speed_limit_raw,

        "angle": angle_raw,

        "theta": theta,

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

    return (
        telemetry,
        raw
    )


# ============================================================
# MODEL LOAD
# ============================================================

def load_model():

    print()
    print(
        "=" * 82
    )

    print(
        "LOADING MODEL"
    )

    print(
        "=" * 82
    )

    if not os.path.isfile(
        MODEL_FILE
    ):

        raise FileNotFoundError(
            "Model not found:\n"
            f"{MODEL_FILE}"
        )

    model = (
        AutonomousDriver()
    )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if (
            "model_state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif (
            "state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        elif (
            "model"
            in checkpoint
        ):

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

    for key, value in (
        state_dict.items()
    ):

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


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def run_model(
    model,
    rgb,
    telemetry
):

    if telemetry.shape != (
        MODEL_TELEMETRY,
    ):

        raise RuntimeError(
            "run_model received telemetry "
            f"shape {telemetry.shape}; "
            f"expected ({MODEL_TELEMETRY},)"
        )

    image_tensor = (
        IMAGE_TRANSFORM(
            rgb
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    telemetry_tensor = (
        torch.from_numpy(
            telemetry
        )
        .float()
        .unsqueeze(0)
        .to(DEVICE)
    )

    output = (
        model(
            image_tensor,
            telemetry_tensor
        )
    )

    if output.shape != (
        1,
        3
    ):

        raise RuntimeError(
            f"Model output shape "
            f"{output.shape}; "
            "expected (1,3)."
        )

    steering = safe_float(
        output[0, 0].item(),
        "steering"
    )

    throttle = safe_float(
        output[0, 1].item(),
        "throttle"
    )

    brake_logit = safe_float(
        output[0, 2].item(),
        "brake logit"
    )

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

    brake_probability = float(
        torch.sigmoid(
            output[0, 2]
        ).item()
    )

    if not math.isfinite(
        brake_probability
    ):

        raise RuntimeError(
            "Brake probability is not finite."
        )

    return (
        steering,
        throttle,
        brake_logit,
        brake_probability
    )


# ============================================================
# APPLY CONTROL
# ============================================================

def apply_control(
    vehicle,
    steering,
    throttle,
    brake_probability
):

    brake = (
        brake_probability
        >= BRAKE_THRESHOLD
    )

    if brake:

        throttle = 0.0

    vehicle.apply_control(
        carla.VehicleControl(
            steer=float(
                steering
            ),

            throttle=float(
                throttle
            ),

            brake=float(
                brake
            ),

            hand_brake=False,

            reverse=False,

            manual_gear_shift=False
        )
    )

    return (
        steering,
        throttle,
        float(brake)
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

    if raw["aim_wp"] is None:

        aim_text = "None"

    else:

        aim_text = (
            f"{raw['aim_wp'][0]:+.2f}, "
            f"{raw['aim_wp'][1]:+.2f}"
        )

    lines = [

        f"Speed: "
        f"{raw['speed']:.2f} m/s",

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

        f"Angle: "
        f"{raw['angle']:+.3f}",

        f"Theta: "
        f"{raw['theta']:+.3f}",

        f"Aim: "
        f"{aim_text}",

        f"Steer: "
        f"{steering:+.3f}",

        f"Throttle: "
        f"{throttle:.3f}",

        f"Brake P: "
        f"{brake_probability:.3f}",
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
            2
        )

        y += 25

    cv2.imshow(
        "PDM-Lite Neural Driver",
        frame
    )


# ============================================================
# TERMINAL DEBUG
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

    transform = (
        vehicle.get_transform()
    )

    yaw = (
        transform.rotation.yaw
    )

    print(
        "\033[2J\033[H",
        end=""
    )

    print(
        "=" * 82
    )

    print(
        "                 NEURAL PDM-LITE DRIVER"
    )

    print(
        "=" * 82
    )

    print(
        "CARLA / SENSOR"
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
        f"Aim WP          : "
        f"{raw['aim_wp']}"
    )

    print(
        "-" * 82
    )

    print(
        "MODEL INPUT — 10 TELEMETRY"
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
    ]

    for i, label in enumerate(
        labels
    ):

        if i in (
            1,
            8
        ):

            print(
                f"{label:<17}: "
                f"{int(telemetry[i])} "
                f"{command_name(telemetry[i])}"
            )

        else:

            print(
                f"{label:<17}: "
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

        print(
            "=" * 82
        )

        print(
            "                 NEURAL PDM-LITE DRIVER"
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
            "=" * 82
        )

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
            ensure_supported_map(
                client
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
        # SYNCHRONOUS MODE
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
        # MODEL
        # ====================================================

        model = load_model()

        # ====================================================
        # SENSOR BLUEPRINTS
        # ====================================================

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
                    x=IMU_X,
                    y=IMU_Y,
                    z=IMU_Z
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
            "  IMU: compass"
        )

        # ====================================================
        # PRIME SENSORS
        # ====================================================

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

        # ====================================================
        # ROUTE
        # ====================================================

        print()

        print(
            "Initializing official "
            "PDM-Lite route planners..."
        )

        route_system = (
            PdmRouteSystem(
                world,
                vehicle
            )
        )

        # ====================================================
        # INITIAL STOP
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
        # DRIVE
        # ====================================================

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
            "No fixed initial navigation command."
        )

        print(
            "Command = official RoutePlanner."
        )

        print(
            "Theta = CARLA IMU -> "
            "official preprocess_compass()."
        )

        print(
            "Angle = PDM-Lite route lookahead."
        )

        print(
            "Telemetry = 10 values."
        )

        print(
            "Brake is NOT an input."
        )

        print(
            "Brake is predicted by the model."
        )

        print(
            "Press Q to stop."
        )

        # ====================================================
        # MAIN LOOP
        # ====================================================

        while True:

            if SHOW_CAMERA:

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord(
                    "q"
                ):

                    print(
                        "\nQ pressed."
                    )

                    break

            # ------------------------------------------------
            # Tick CARLA
            # ------------------------------------------------

            frame = (
                world.tick()
            )

            # ------------------------------------------------
            # Sensors
            # ------------------------------------------------

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
            # Camera
            # ------------------------------------------------

            rgb = image_to_rgb(
                camera_data
            )

            # ------------------------------------------------
            # IMU / theta
            # ------------------------------------------------

            compass, theta = (
                get_pdm_theta(
                    imu_data
                )
            )

            # ------------------------------------------------
            # Route update
            # ------------------------------------------------

            route_system.update()

            # ------------------------------------------------
            # Telemetry
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
            # Model
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
            # Control
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
            # Debug
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
            # Camera
            # ------------------------------------------------

            show_camera(
                rgb,
                telemetry,
                raw,
                applied_steering,
                applied_throttle,
                brake_probability
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
        # Brake vehicle before destruction
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
        # Camera
        # ----------------------------------------------------

        if camera is not None:

            try:
                camera.stop()
            except Exception:
                pass

            try:
                camera.destroy()
            except Exception:
                pass

        # ----------------------------------------------------
        # IMU
        # ----------------------------------------------------

        if imu is not None:

            try:
                imu.stop()
            except Exception:
                pass

            try:
                imu.destroy()
            except Exception:
                pass

        # ----------------------------------------------------
        # Restore CARLA settings
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
        # Camera windows
        # ----------------------------------------------------

        try:

            cv2.destroyAllWindows()

        except Exception:

            pass

        # ----------------------------------------------------
        # Working directory
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
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()