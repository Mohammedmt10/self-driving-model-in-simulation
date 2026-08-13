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
    """

    all_paths = []

    for base_path in base_paths:

        route_folders = glob.glob(os.path.join(base_path,"Route*"))

        for route_folder in route_folders:

            rgb_folder = os.path.join(route_folder, "rgb")

            measurements_folder = os.path.join(route_folder, "measurements")

            if not os.path.isdir(rgb_folder):
                continue

            for img_file in os.listdir(rgb_folder):
                base_name = os.path.splitext(img_file)[0]

                img_path = os.path.join(rgb_folder, img_file)

                json_path = os.path.join(measurements_folder, base_name + ".json.gz")

                if os.path.isfile(json_path):
                    all_paths.append((img_path, json_path))

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
        value = COMMAND_MAP.get(value.upper(), 4)

    else:
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 4

    return max(1, min(value, 6))


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
            transforms.Resize((200,400)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.master_list)

    def __getitem__(self, idx):

        img_path, json_path = (self.master_list[idx])

        # ====================================================
        # LOAD JSON
        # ====================================================

        with gzip.open(json_path, "rt", encoding="utf-8") as f:
            telemetry = json.load(f)

        # ====================================================
        # TARGETS
        # ====================================================

        steer = float(telemetry.get("steer") or 0.0)

        throttle = float(telemetry.get("throttle") or 0.0)

        control_brake = float(bool(telemetry.get("control_brake") or False))

        # ====================================================
        # CONTINUOUS INPUTS
        # ====================================================

        speed = float(telemetry.get("speed") or 0.0)

        speed_limit = float(telemetry.get("speed_limit") or 50.0)

        object_distance = float(telemetry.get("speed_reduced_by_obj_distance") or 0.0)

        angle = float(telemetry.get("angle") or 0.0)

        theta = float(telemetry.get("theta") or 0.0)

        junction = float(bool(telemetry.get("junction") or False))

        # ====================================================
        # COMMANDS
        # ====================================================

        command = parse_command(telemetry.get("command"))

        next_command = parse_command(telemetry.get("next_command"))

        # ====================================================
        # OBJECT TYPE
        # ====================================================

        object_type = OBJECT_TYPE_MAP.get(telemetry.get("speed_reduced_by_obj_type"),0)

        object_type = max(0, min(object_type,4))

        # ====================================================
        # OBJECT ID
        # ====================================================

        try:
            object_id = int(telemetry.get("speed_reduced_by_obj_id") or -1)

        except (TypeError, ValueError):
            object_id = -1

        # ====================================================
        # NORMALIZATION
        #
        # SAME NORMALIZATION USED IN OUR 11-FEATURE PIPELINE
        # ====================================================

        speed = np.clip(speed / 30.0, 0.0, 2.0)

        speed_limit = np.clip(speed_limit / 30.0, 0.0, 2.0)

        object_distance = np.clip(object_distance / 50.0, 0.0, 1.0)

        angle = np.clip(angle / np.pi, -1.0, 1.0)

        theta = np.clip(theta / np.pi, -1.0, 1.0)

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
        # 7  brake
        # 8  angle
        # 9  next_command
        # 10 theta
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
            ],
            dtype=torch.float32
        )

        # ====================================================
        # IMAGE
        # ====================================================

        image = mpimg.imread(img_path)

        image = np.array(image, copy=True)

        image_tensor = (self.transform(image))

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
#
# V4 CHANGE:
#
# The dataset is approximately:
#
#   91.87% straight
#    2.81% left
#    3.74% right
#    0.58% strong left
#    0.99% strong right
#
# We use sqrt inverse-frequency weighting instead of
# full inverse-frequency weighting to partially rebalance
# the distribution without forcing 20% of training to be
# each bucket.
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
    # Assign bucket
    # --------------------------------------------------------

    for _, json_path in master_list:

        with gzip.open(json_path, "rt", encoding="utf-8") as f:
            telemetry = json.load(f)

        steer = float(telemetry.get("steer") or 0.0)

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
    # sqrt inverse-frequency
    # --------------------------------------------------------

    sample_weights = []

    for bucket in buckets:

        count = counts[bucket]
        weight = 1.0 / math.sqrt(max(count, 1))

        sample_weights.append(weight)

    return (
        sample_weights,
        counts
    )


# ============================================================
# OPTIONAL DATASET SUMMARY
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

        with gzip.open(json_path, "rt", encoding="utf-8") as f:
            telemetry = json.load(f)

        steer = float(telemetry.get("steer") or 0.0)

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

    total = max(len(master_list), 1)

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

        percentage = (100.0* count/ total)

        print(
            f"{name:>13}: "
            f"{count:6d} "
            f"({percentage:6.2f}%)"
)

    return counts