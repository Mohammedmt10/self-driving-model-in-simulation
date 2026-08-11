import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import glob
import gzip
import json
import matplotlib.image as mpimg

def build_path_index(base_paths, keep_straight_ratio=0.2):
    """Step 1: Just get the file paths."""
    all_paths = []

    straight_count = 0
    turn_count = 0

    for base_path in base_paths:
        route_folders = glob.glob(os.path.join(base_path, "Route*"))
        for route_folder in route_folders:
            rgb_folder = os.path.join(route_folder, "rgb")
            measurements_folder = os.path.join(route_folder, "measurements")
            
            if not os.path.exists(rgb_folder): continue
                
            for img_file in os.listdir(rgb_folder):
                base_name = os.path.splitext(img_file)[0]
                json_filename = base_name + ".json.gz"
                
                img_path = os.path.join(rgb_folder, img_file)
                json_path = os.path.join(measurements_folder, json_filename)
                
                if os.path.exists(json_path):
                    all_paths.append((img_path, json_path))

                    with gzip.open(json_path, 'rt', encoding='utf-8') as f:
                        telemetry = json.load(f)
                    
                    steer = float(telemetry.get('steer', 0.0))
                    
                    # If steering is near zero, it's a straight lane
                    if abs(steer) < 0.05:
                        # Drop a massive percentage of them
                        if random.random() < keep_straight_ratio:
                            all_paths.append((img_path, json_path))
                            straight_count += 1
                    else:
                        # It is a turn! Keep 100% of these.
                        all_paths.append((img_path, json_path))
                        turn_count += 1
                
    return all_paths


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


class CarlaDataset(Dataset):
    def __init__(self, master_list):
        self.master_list = master_list
        self.transform = transforms.Compose([
            transforms.ToPILImage(),             # Ensures it's a PIL image if starting from a numpy array
            transforms.Resize((200, 200)),       # Resize first (matching your 200x200 model input)
            transforms.ToTensor(),               # Convert to tensor last
        ])

    def __len__(self):
        # Tells PyTorch exactly how many frames are in your dataset
        return len(self.master_list)

    def __getitem__(self, idx):
        # PyTorch will call this function to grab one specific frame
        img_path, json_path = self.master_list[idx]

        # 1. Load Telemetry
        with gzip.open(json_path, 'rt', encoding='utf-8') as f:
            telemetry = json.load(f)

        speed = float(telemetry.get('speed') or 0.0)
        command = float(telemetry.get('command') or 4.0)
        steer = float(telemetry.get('steer') or 0.0)
        throttle = float(telemetry.get('throttle') or 0.0)
        control_brake = float(telemetry.get('control_brake') or 0.0)
        brake = bool(telemetry.get('brake') or 0.0)
        junction = bool(telemetry.get('junction') or False)
        speed_reduced_by_obj_type = OBJECT_TYPE_MAP.get(telemetry.get('speed_reduced_by_obj_type'), 0)
        speed_reduced_by_obj_id = int(telemetry.get('speed_reduced_by_obj_id') or -1)
        speed_reduced_by_obj_distance = float(telemetry.get('speed_reduced_by_obj_distance') or 0.0)
        speed_limit = float(telemetry.get('speed_limit') or 50.0)

        # 2. Load Image
        image = mpimg.imread(img_path)
        image = np.array(image, copy=True)

        top = int(0.15 * image.shape[0])  # Crop top 35%
        bottom = int(0.10 * image.shape[0])  # Crop bottom
        image = image[top:-bottom, :, :]

        # Convert to PyTorch Tensor format
        image_tensor = self.transform(image) 

        # 3. Bundle the Inputs and Targets
        # We convert the extra inputs and targets into PyTorch tensors
        telemetry_tensor = torch.tensor([speed, command, junction, speed_reduced_by_obj_type, speed_reduced_by_obj_id, speed_reduced_by_obj_distance, speed_limit, control_brake], dtype=torch.float32)
        target_tensor = torch.tensor([steer, throttle, brake], dtype=torch.float32)

        return image_tensor, telemetry_tensor, target_tensor