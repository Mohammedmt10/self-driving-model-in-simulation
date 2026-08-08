import os
import glob
import gzip
import json

def master_dataset_index(base_paths):
    """
    Create a master index of all images in the dataset.
    :return: A list of all image file paths.
    """
    dataset_index = []

    for base_path in base_paths:
        route_folders = glob.glob(os.path.join(base_path, "Route*"))

        for route_folder in route_folders:
            rgb_folder = os.path.join(route_folder, "rgb")
            measurements_folder = os.path.join(route_folder, "measurements")

            images = sorted(os.listdir(rgb_folder))

            for img_file in images:
                base_name = os.path.splitext(img_file)[0]

                json_filename = os.path.splitext(base_name)[0] + ".json.gz"
                json_path = os.path.join(measurements_folder, json_filename)

                if os.path.exists(json_path):
                    with gzip.open(json_path, 'rt') as f:
                        telemetry = json.load(f)

                        steering_angle = telemetry.get("steer", None)
                        throttle = telemetry.get("throttle", None)
                        brake = float(telemetry.get("brake", None))

                        print(f"Image: {img_file}, Steering Angle: {steering_angle}, Throttle: {throttle}, Brake: {brake}")


master_dataset_index(['./DynamicObjectCrossing'])