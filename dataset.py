import os
import glob
import gzip
import json
import math

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.image as mpimg


# ============================================================
# PATH INDEX
# ============================================================

def build_path_index(base_paths):

    """
    Build:

        [(image_path, measurement_path), ...]

    from:

        Route*/rgb/*
        Route*/measurements/*.json.gz
    """

    all_paths = []

    for base_path in base_paths:

        route_folders = glob.glob(
            os.path.join(base_path, "Route*")
        )

        for route_folder in route_folders:

            rgb_folder = os.path.join(
                route_folder,
                "rgb"
            )

            measurements_folder = os.path.join(
                route_folder,
                "measurements"
            )

            if not os.path.isdir(rgb_folder):
                continue

            if not os.path.isdir(measurements_folder):
                continue

            for img_file in os.listdir(rgb_folder):

                base_name = os.path.splitext(img_file)[0]

                img_path = os.path.join(
                    rgb_folder,
                    img_file
                )

                json_path = os.path.join(
                    measurements_folder,
                    base_name + ".json.gz"
                )

                if os.path.isfile(json_path):

                    all_paths.append(
                        (
                            img_path,
                            json_path
                        )
                    )

    return all_paths


# ============================================================
# OBJECT TYPE MAP
# ============================================================

OBJECT_TYPE_MAP = {

    "None": 0,
    None: 0,

    "traffic.stop": 1,
    "traffic.traffic_light": 2,

    "vehicle.audi.tt": 3,
    "vehicle.carlamotors.firetruck": 3,
    "vehicle.chevrolet.impala": 3,
    "vehicle.diamondback.century": 3,
    "vehicle.dodge.charger_2020": 3,
    "vehicle.dodge.charger_police": 3,
    "vehicle.dodge.charger_police_2020": 3,
    "vehicle.ford.ambulance": 3,
    "vehicle.ford.mustang": 3,
    "vehicle.lincoln.mkz_2017": 3,
    "vehicle.lincoln.mkz_2020": 3,
    "vehicle.mercedes.coupe_2020": 3,
    "vehicle.mini.cooper_s_2021": 3,
    "vehicle.nissan.patrol_2021": 3,

    "walker.pedestrian.0001": 4,
    "walker.pedestrian.0002": 4,
    "walker.pedestrian.0005": 4,
    "walker.pedestrian.0010": 4,
    "walker.pedestrian.0014": 4,
    "walker.pedestrian.0020": 4,
    "walker.pedestrian.0030": 4,
    "walker.pedestrian.0036": 4,
    "walker.pedestrian.0038": 4,
    "walker.pedestrian.0044": 4,
    "walker.pedestrian.0047": 4,
}


# ============================================================
# COMMAND MAP
# ============================================================

COMMAND_MAP = {

    "LEFT": 1,
    "RIGHT": 2,
    "STRAIGHT": 3,
    "LANEFOLLOW": 4,
    "CHANGE_LANE_LEFT": 5,
    "CHANGE_LANE_RIGHT": 6,
}


def parse_command(value):

    if isinstance(value, str):

        value = COMMAND_MAP.get(
            value.upper(),
            4
        )

    else:

        try:
            value = int(value)

        except (TypeError, ValueError):

            value = 4

    return max(
        1,
        min(value, 6)
    )


# ============================================================
# LATERAL DISTANCE FROM EGO-LOCAL ROUTE
# ============================================================

def calculate_lateral_distance(telemetry):

    """
    Calculate signed lateral distance between the ego vehicle
    and the planned route.

    IMPORTANT:

    PDM-Lite / CARLA-Garage stores `route` in the ego vehicle's
    LOCAL coordinate frame.

    CARLA coordinate convention:

        X -> forward
        Y -> right

    Therefore:

        route = [[x1, y1], [x2, y2], ...]

    already describes where the planned route lies relative
    to the ego vehicle.

    We DO NOT use:

        pos_global
        route_yaw
        route_center
        road_center
        lane_center
        theta

    for this calculation.

    The ego vehicle is at:

        (0, 0)

    in this coordinate frame.

    We use the first route segment and calculate the signed
    perpendicular distance from the ego position to that
    segment's supporting line.

    Sign convention:

        negative -> ego is LEFT of route
        positive -> ego is RIGHT of route
    """

    route = telemetry.get("route")

    if route is None:
        return 0.0

    if not isinstance(route, (list, tuple)):
        return 0.0

    if len(route) < 2:
        return 0.0

    # --------------------------------------------------------
    # Find first two valid route points.
    # --------------------------------------------------------

    p1 = None
    p2 = None

    for i in range(len(route) - 1):

        try:

            x1 = float(route[i][0])
            y1 = float(route[i][1])

            x2 = float(route[i + 1][0])
            y2 = float(route[i + 1][1])

        except (
            TypeError,
            ValueError,
            IndexError
        ):

            continue

        dx = x2 - x1
        dy = y2 - y1

        length = math.sqrt(
            dx * dx +
            dy * dy
        )

        # Ignore duplicate / invalid points.
        if length < 1e-6:
            continue

        p1 = (x1, y1)
        p2 = (x2, y2)

        break

    if p1 is None or p2 is None:
        return 0.0

    x1, y1 = p1
    x2, y2 = p2

    # --------------------------------------------------------
    # Route tangent.
    # --------------------------------------------------------

    dx = x2 - x1
    dy = y2 - y1

    length = math.sqrt(
        dx * dx +
        dy * dy
    )

    tx = dx / length
    ty = dy / length

    # --------------------------------------------------------
    # CARLA right vector in the LOCAL XY plane.
    #
    # For a route pointing straight forward:
    #
    # tangent = (1, 0)
    #
    # right = (0, 1)
    #
    # which is exactly CARLA's +Y direction.
    # --------------------------------------------------------

    right_x = -ty
    right_y = tx

    # --------------------------------------------------------
    # Ego position in its own local coordinate frame.
    # --------------------------------------------------------

    ego_x = 0.0
    ego_y = 0.0

    # --------------------------------------------------------
    # Vector from route point to ego.
    # --------------------------------------------------------

    relative_x = ego_x - x1
    relative_y = ego_y - y1

    # --------------------------------------------------------
    # Signed lateral distance.
    #
    # Negative:
    #     ego is left of route.
    #
    # Positive:
    #     ego is right of route.
    # --------------------------------------------------------

    lateral_distance = (
        relative_x * right_x +
        relative_y * right_y
    )

    return float(lateral_distance)


# ============================================================
# DATASET
#
# EXACTLY 11 TELEMETRY VALUES
# ============================================================

class CarlaDataset(Dataset):

    def __init__(self, master_list):

        self.master_list = master_list

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((200, 400)),
            transforms.ToTensor(),
        ])

    def __len__(self):

        return len(self.master_list)

    def __getitem__(self, idx):

        img_path, json_path = self.master_list[idx]

        # ====================================================
        # LOAD JSON
        # ====================================================

        with gzip.open(
            json_path,
            "rt",
            encoding="utf-8"
        ) as f:

            telemetry = json.load(f)

        # ====================================================
        # TARGETS
        # ====================================================

        steer = float(
            telemetry.get("steer") or 0.0
        )

        throttle = float(
            telemetry.get("throttle") or 0.0
        )

        control_brake = float(
            bool(
                telemetry.get("control_brake") or False
            )
        )

        # ====================================================
        # CONTINUOUS INPUTS
        # ====================================================

        speed = float(
            telemetry.get("speed") or 0.0
        )

        speed_limit = float(
            telemetry.get("speed_limit") or 50.0
        )

        object_distance = float(
            telemetry.get(
                "speed_reduced_by_obj_distance"
            ) or 0.0
        )

        # ----------------------------------------------------
        # DO NOT CHANGE THESE.
        # ----------------------------------------------------

        angle = float(
            telemetry.get("angle") or 0.0
        )

        theta = float(
            telemetry.get("theta") or 0.0
        )

        junction = float(
            bool(
                telemetry.get("junction") or False
            )
        )

        # ====================================================
        # COMMANDS
        # ====================================================

        command = parse_command(
            telemetry.get("command")
        )

        next_command = parse_command(
            telemetry.get("next_command")
        )

        # ====================================================
        # OBJECT TYPE
        # ====================================================

        object_type = OBJECT_TYPE_MAP.get(
            telemetry.get(
                "speed_reduced_by_obj_type"
            ),
            0
        )

        object_type = max(
            0,
            min(object_type, 4)
        )

        # ====================================================
        # OBJECT ID
        # ====================================================

        try:

            object_id = int(
                telemetry.get(
                    "speed_reduced_by_obj_id"
                ) or -1
            )

        except (
            TypeError,
            ValueError
        ):

            object_id = -1

        # ====================================================
        # LATERAL DISTANCE
        # ====================================================

        lateral_distance = calculate_lateral_distance(
            telemetry
        )

        # ====================================================
        # NORMALIZATION
        # ====================================================

        # Speed:
        # 30 m/s reference.
        speed = np.clip(
            speed / 30.0,
            0.0,
            2.0
        )

        # Speed limit:
        # 30 m/s reference.
        speed_limit = np.clip(
            speed_limit / 30.0,
            0.0,
            2.0
        )

        # Object distance:
        # 50 m reference.
        object_distance = np.clip(
            object_distance / 50.0,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # EXISTING ANGLE NORMALIZATION.
        # DO NOT CHANGE.
        # ----------------------------------------------------

        angle = np.clip(
            angle / np.pi,
            -1.0,
            1.0
        )

        # ----------------------------------------------------
        # EXISTING THETA NORMALIZATION.
        # DO NOT CHANGE.
        # ----------------------------------------------------

        theta = np.clip(
            theta / np.pi,
            -1.0,
            1.0
        )

        # ----------------------------------------------------
        # LATERAL DISTANCE NORMALIZATION.
        #
        # 2 meters = 1.0
        #
        # -2 m -> -1
        #  0 m ->  0
        # +2 m -> +1
        #
        # Values outside +/-2 m are clipped.
        # ----------------------------------------------------

        lateral_distance = np.clip(
            lateral_distance / 2.0,
            -1.0,
            1.0
        )

        # ====================================================
        # EXACT 11 INPUTS
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
        # 10 lateral_distance
        # ====================================================

        telemetry_tensor = torch.tensor(
            [
                speed,
                command,
                junction,
                object_type,
                object_id,
                object_distance,
                speed_limit,
                angle,
                next_command,
                theta,
                lateral_distance,
            ],
            dtype=torch.float32
        )

        # ====================================================
        # IMAGE
        # ====================================================

        image = mpimg.imread(
            img_path
        )

        image = np.array(
            image,
            copy=True
        )

        image_tensor = self.transform(
            image
        )

        # ====================================================
        # TARGET
        # ====================================================

        target_tensor = torch.tensor(
            [
                steer,
                throttle,
                control_brake,
            ],
            dtype=torch.float32
        )

        return (
            image_tensor,
            telemetry_tensor,
            target_tensor
        )


# ============================================================
# STEERING SAMPLING WEIGHTS
# ============================================================

def get_steering_sampling_weights(master_list):

    buckets = []

    counts = {
        "strong_left": 0,
        "left": 0,
        "straight": 0,
        "right": 0,
        "strong_right": 0,
    }

    # --------------------------------------------------------
    # Assign steering bucket.
    # --------------------------------------------------------

    for _, json_path in master_list:

        with gzip.open(
            json_path,
            "rt",
            encoding="utf-8"
        ) as f:

            telemetry = json.load(f)

        steer = float(
            telemetry.get("steer") or 0.0
        )

        if steer < -0.50:

            bucket = "strong_left"

        elif steer < -0.15:

            bucket = "left"

        elif steer <= 0.15:

            bucket = "straight"

        elif steer <= 0.50:

            bucket = "right"

        else:

            bucket = "strong_right"

        buckets.append(bucket)

        counts[bucket] += 1

    # --------------------------------------------------------
    # sqrt inverse-frequency weighting.
    # --------------------------------------------------------

    sample_weights = []

    for bucket in buckets:

        count = counts[bucket]

        weight = (
            1.0 /
            math.sqrt(
                max(count, 1)
            )
        )

        sample_weights.append(weight)

    return (
        sample_weights,
        counts
    )


# ============================================================
# STEERING DISTRIBUTION
# ============================================================

def summarize_steering_distribution(master_list):

    counts = {
        "strong_left": 0,
        "left": 0,
        "straight": 0,
        "right": 0,
        "strong_right": 0,
    }

    for _, json_path in master_list:

        with gzip.open(
            json_path,
            "rt",
            encoding="utf-8"
        ) as f:

            telemetry = json.load(f)

        steer = float(
            telemetry.get("steer") or 0.0
        )

        if steer < -0.50:

            counts["strong_left"] += 1

        elif steer < -0.15:

            counts["left"] += 1

        elif steer <= 0.15:

            counts["straight"] += 1

        elif steer <= 0.50:

            counts["right"] += 1

        else:

            counts["strong_right"] += 1

    total = max(
        len(master_list),
        1
    )

    print()
    print(
        "Steering distribution:"
    )

    for name in (
        "strong_left",
        "left",
        "straight",
        "right",
        "strong_right"
    ):

        count = counts[name]

        percentage = (
            100.0 *
            count /
            total
        )

        print(
            f"{name:>13}: "
            f"{count:6d} "
            f"({percentage:6.2f}%)"
        )

    return counts


# ============================================================
# LATERAL DISTANCE DEBUG
# ============================================================

def debug_lateral_distance(json_path):

    """
    Useful for checking the actual route-based lateral
    distance on one measurement file.
    """

    with gzip.open(
        json_path,
        "rt",
        encoding="utf-8"
    ) as f:

        telemetry = json.load(f)

    raw_distance = calculate_lateral_distance(
        telemetry
    )

    normalized_distance = np.clip(
        raw_distance / 2.0,
        -1.0,
        1.0
    )

    print()
    print("=" * 70)
    print("LATERAL DISTANCE DEBUG")
    print("=" * 70)

    print(
        f"Route points       : "
        f"{len(telemetry.get('route', []))}"
    )

    print(
        f"Raw lateral distance: "
        f"{raw_distance:.6f} m"
    )

    print(
        f"Normalized          : "
        f"{normalized_distance:.6f}"
    )

    print(
        "Normalization       : "
        "distance / 2.0"
    )

    print("=" * 70)

    return (
        raw_distance,
        normalized_distance
    )