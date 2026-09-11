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


def _parse_route_num(route_folder_name):
    """Extract the route number from a folder name like 'Route12_Rep0'."""
    try:
        return int(route_folder_name.split("_")[0].replace("Route", ""))
    except (ValueError, IndexError):
        return -1


def _build_route_pairs(route_folder):
    """Return [(image_path, measurement_path), ...] for one route folder."""
    rgb_folder = os.path.join(route_folder, "rgb")
    measurements_folder = os.path.join(route_folder, "measurements")
    if not os.path.isdir(rgb_folder) or not os.path.isdir(measurements_folder):
        return []

    pairs = []
    for img_file in os.listdir(rgb_folder):
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(rgb_folder, img_file)
        json_path = os.path.join(measurements_folder, base_name + ".json.gz")
        if os.path.isfile(json_path):
            pairs.append((img_path, json_path))
    return pairs


def build_route_split(towns_dir, val_fraction=0.2):
    """
    Route-based train/validation split across all towns.

    For each (town, scenario) group, the Route* folders are sorted by
    route number and split ~80/20 by route: the first 80% of routes go
    to training, the last 20% to validation. Validation therefore holds
    out entire *routes* (unseen road paths) rather than entire towns, so
    the model still trains on every town's roads.

    Returns (train_list, val_list), each a list of
    (image_path, measurement_path) tuples.
    """
    train_list = []
    val_list = []

    for town in sorted(os.listdir(towns_dir)):
        town_dir = os.path.join(towns_dir, town)
        if not os.path.isdir(town_dir):
            continue
        data_dir = os.path.join(town_dir, "data")
        if not os.path.isdir(data_dir):
            continue

        for scenario in sorted(os.listdir(data_dir)):
            scenario_dir = os.path.join(data_dir, scenario)
            if not os.path.isdir(scenario_dir):
                continue

            route_folders = sorted(
                (
                    p for p in os.listdir(scenario_dir)
                    if p.startswith("Route")
                    and os.path.isdir(os.path.join(scenario_dir, p))
                ),
                key=_parse_route_num,
            )

            if not route_folders:
                continue

            n_val = int(round(len(route_folders) * val_fraction))
            # Keep at least one route for training when possible.
            if len(route_folders) > 1:
                n_val = min(max(n_val, 1), len(route_folders) - 1)
            else:
                n_val = 0
            n_train = len(route_folders) - n_val

            for route in route_folders[:n_train]:
                train_list.extend(
                    _build_route_pairs(os.path.join(scenario_dir, route))
                )
            for route in route_folders[n_train:]:
                val_list.extend(
                    _build_route_pairs(os.path.join(scenario_dir, route))
                )

    return train_list, val_list


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

    We find the NEAREST route segment and compute the signed
    perpendicular distance from the ego position to that
    segment's supporting line. This matches how the driver
    (drive.py -> CarlaRouteState.calculate_lateral_distance)
    computes lateral distance at inference time.

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
    # Nearest-range check (same convention as drive.py).
    # --------------------------------------------------------

    eps = 1e-9

    best_abs = float("inf")
    best_lateral = 0.0

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
        if length < eps:
            continue

        # ----------------------------------------------------
        # Route tangent and CARLA right vector in the LOCAL XY
        # plane.  For a route pointing straight forward:
        #
        #     tangent = (1, 0)
        #     right   = (0, 1)   (= CARLA +Y)
        # ----------------------------------------------------

        tx = dx / length
        ty = dy / length

        right_x = -ty
        right_y = tx

        # ----------------------------------------------------
        # Signed perpendicular distance from the ego (origin,
        # 0,0) to this segment's supporting line.
        #
        # Negative -> left of route.
        # Positive -> right of route.
        # ----------------------------------------------------

        lateral = (
            (0.0 - x1) * right_x +
            (0.0 - y1) * right_y
        )

        if abs(lateral) < best_abs:

            best_abs = abs(lateral)
            best_lateral = lateral

    return float(best_lateral)


# ============================================================
# IMAGE
#
# Single source of truth for how training images are loaded.
# IMAGE_SPEC documents the pipeline; CarlaDataset.transform is
# built FROM it, so the code and its description cannot drift.
#
# Pipeline (must match drive.py run_model / IMAGE_TRANSFORM):
#
#   [stored PNG, RGB, 512 x 1024]  (written by the CARLA camera)
#       -> mpimg.imread  -> float32 [0, 1] (H, W, 3)
#       -> drop top 30%            (sky / buildings / trees)
#       -> ToPILImage
#       -> grayscale (luminance)
#       -> resize to 480 x 240     (H, W)
#       -> ToTensor -> float32 tensor (1, 240, 480), values in [0, 1]
#       -> NO mean/std normalization
# ============================================================

IMAGE_SPEC = {
    # --- input ---
    "input": {
        "channels": "RGB",
        "dtype": "float32 (mpimg.imread)",
        "range": [0.0, 1.0],
        "height": 512,
        "width": 1024,
    },
    # --- crop ---
    "crop": {
        "ratio": 0.3,
        "expr": "int(0.3 * image.shape[0])",
        "px": 153,
        "note": (
            "top of the frame is removed before resize, "
            "exactly like drive.py"
        ),
    },
    # --- grayscale ---
    "grayscale": True,
    "grayscale_formula": (
        "luminance: 0.299 R + 0.587 G + 0.114 B"
    ),
    # --- resize ---
    "resize": (240, 480),   # (H, W)
    "resize_interpolation": (
        "BILINEAR (torchvision Resize default)"
    ),
    # --- output tensor ---
    "output": {
        "channels": 1,
        "shape": (1, 240, 480),
        "dtype": "float32",
        "range": [0.0, 1.0],
        "normalization": "none (raw [0,1])",
    },
    # --- reference statistics ---
    "note": (
        "IMG_MEAN / IMG_STD (0.4020 / 0.1886) are whole-dataset "
        "statistics for reference only, NOT applied here — the "
        "model trains on raw [0,1] grayscale."
    ),
}


def report_image_spec(title="DATASET LOAD IMAGE SPEC"):

    """
    Print the exact image format the dataset produces, so it can
    be compared against the driver inference path without reading
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
# DATASET
#
# TELEMETRY VECTOR: 16 VALUES
#
#   indices 0..7   original 8 telemetry inputs (consumed by the
#                  model's AutonomousDriver.forward)
#   indices 8..15  object / hazard measurements appended from the
#                  measurement JSON (not yet consumed by the model
#                  — added so explicit object-distance, hazard and
#                  object-type signals are available for the next
#                  model change)
# ============================================================

class CarlaDataset(Dataset):

    def __init__(self, master_list):

        self.master_list = master_list

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize(IMAGE_SPEC["resize"]),
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
        # LATERAL DISTANCE
        # ====================================================

        lateral_distance = calculate_lateral_distance(
            telemetry
        )

        # ====================================================
        # OBJECT / HAZARD MEASUREMENTS
        #
        # New telemetry columns (indices 8..15), appended after
        # the original 8. These expose explicit object-distance,
        # hazard and object-type signals to the braking / evasion
        # policy.
        # ====================================================

        # Distance (m) to the object currently causing the ego to
        # reduce speed. Absent when nothing is slowing the ego;
        # treated as "far away" (500 m) so it normalizes to the
        # max value ("no obstructing object").
        obj_dist = telemetry.get(
            "speed_reduced_by_obj_distance"
        )

        if obj_dist is None:

            obj_dist = 500.0

        obj_dist = float(
            obj_dist
        )

        # Binary hazard flags, cast to 0.0 / 1.0.
        vehicle_hazard = float(
            bool(
                telemetry.get("vehicle_hazard") or False
            )
        )

        light_hazard = float(
            bool(
                telemetry.get("light_hazard") or False
            )
        )

        walker_hazard = float(
            bool(
                telemetry.get("walker_hazard") or False
            )
        )

        stop_sign_hazard = float(
            bool(
                telemetry.get("stop_sign_hazard") or False
            )
        )

        stop_sign_close = float(
            bool(
                telemetry.get("stop_sign_close") or False
            )
        )

        walker_close = float(
            bool(
                telemetry.get("walker_close") or False
            )
        )

        # Object type / name, collapsed to a category id:
        #
        #   0  none (no speed-reducing object)
        #   1  vehicle
        #   2  walker / pedestrian
        #   3  traffic light
        #   4  stop sign
        #
        # The raw string (e.g. "vehicle.lincoln.mkz_2017") has 27
        # distinct values whose model-specific suffixes ("_2017" vs
        # "_2020") carry no policy signal, so we keep only the
        # top-level category. Kept as a small integer (like command)
        # so the model can embed it.
        obj_type_raw = (
            telemetry.get("speed_reduced_by_obj_type") or ""
        )

        if not obj_type_raw:

            obj_type = 0

        elif obj_type_raw.startswith("vehicle"):

            obj_type = 1

        elif obj_type_raw.startswith("walker"):

            obj_type = 2

        elif obj_type_raw == "traffic.traffic_light":

            obj_type = 3

        else:

            # "traffic.stop" and anything else.
            obj_type = 4

        obj_type = float(obj_type)

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

        # ----------------------------------------------------
        # OBJECT DISTANCE NORMALIZATION.
        #
        # 50 m reference, clipped to [0, 2]:
        #
        #   0   m -> 0.0   (object right in front)
        #   50  m -> 1.0
        #   >=100 m -> 2.0 (far away / no obstructing object)
        #
        # A missing value (no speed-reducing object) is treated
        # as 500 m, which lands at the clipped max of 2.0.
        # ----------------------------------------------------

        obj_dist = np.clip(
            obj_dist / 50.0,
            0.0,
            2.0
        )

        # ====================================================
        # TELEMETRY INPUT
        #
        # Original 8 (run through the model today):
        #   0  speed
        #   1  command
        #   2  junction
        #   3  speed_limit
        #   4  angle
        #   5  next_command
        #   6  theta
        #   7  lateral_distance
        #
        # Appended object / hazard measurements:
        #   8  obj_dist           (normalized; 2.0 = far/none)
        #   9  vehicle_hazard
        #   10 light_hazard
        #   11 walker_hazard
        #   12 stop_sign_hazard
        #   13 stop_sign_close
        #   14 walker_close
        #   15 obj_type           (category id 0-4, NOT normalized:
        #                         embed like command)
        # ====================================================

        telemetry_tensor = torch.tensor(
            [
                speed,
                command,
                junction,
                speed_limit,
                angle,
                next_command,
                theta,
                lateral_distance,
                obj_dist,
                vehicle_hazard,
                light_hazard,
                walker_hazard,
                stop_sign_hazard,
                stop_sign_close,
                walker_close,
                obj_type,
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

        # Drop the top 30% (sky / buildings / trees) to remove
        # route-correlated scene context that causes overfitting.
        # Must stay in sync with IMAGE_SPEC above and with
        # drive.py -> IMAGE_CROP_RATIO.
        image = image[int(0.3 * image.shape[0]):, :]

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
# LOSS NORMALIZATION WEIGHTS (BASELINE LOSS)
# ============================================================

def compute_loss_weights(master_list):

    """
    Compute per-output loss-normalization weights directly from
    the training data -- no assumed scales.

    Each weight is the loss a trivial constant predictor achieves
    on that task:

        steer    -> MSE of predicting the mean steer
                    (= variance of steer)
        throttle -> MSE of predicting the mean throttle
                    (= variance of throttle)
        brake    -> cross-entropy of predicting the marginal
                    class distribution (= entropy)

    Dividing each raw task loss by its weight makes every task
    contribute ~1.0 to the total at the start of training.
    """

    steer_sum = 0.0
    steer_sq_sum = 0.0
    throttle_sum = 0.0
    throttle_sq_sum = 0.0
    brake_ones = 0
    n = 0

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

        throttle = float(
            telemetry.get("throttle") or 0.0
        )

        brake = float(
            bool(
                telemetry.get("control_brake") or False
            )
        )

        steer_sum += steer
        steer_sq_sum += steer * steer
        throttle_sum += throttle
        throttle_sq_sum += throttle * throttle
        brake_ones += brake
        n += 1

    n = max(n, 1)

    steer_mean = steer_sum / n
    throttle_mean = throttle_sum / n

    # E[y^2] - E[y]^2 == MSE of predicting the mean.
    steer_var = max(
        steer_sq_sum / n - steer_mean * steer_mean,
        1e-9
    )

    throttle_var = max(
        throttle_sq_sum / n - throttle_mean * throttle_mean,
        1e-9
    )

    # Binary cross-entropy of predicting the marginal distribution.
    p_brake = min(
        max(brake_ones / n, 1e-9),
        1.0 - 1e-9
    )

    brake_entropy = -(
        p_brake * math.log(p_brake) +
        (1.0 - p_brake) * math.log(1.0 - p_brake)
    )

    return {
        "steer": steer_var,
        "throttle": throttle_var,
        "brake": brake_entropy,
        "counts": {
            "n": n,
            "brake_ones": brake_ones,
            "p_brake": p_brake,
        },
    }


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
