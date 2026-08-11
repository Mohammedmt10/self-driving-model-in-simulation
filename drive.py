import sys

# ============================================================
# CARLA PYTHONAPI PATH
# ============================================================

CARLA_PYTHONAPI = "/home/mohammedtajir/carla-simulator/PythonAPI/carla"

if CARLA_PYTHONAPI not in sys.path:
    sys.path.append(CARLA_PYTHONAPI)


import carla
import math
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import queue

from model import AutonomousDriver

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "carla_driver_pytorch_v1_best.pth"

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200

CAMERA_WIDTH = 400
CAMERA_HEIGHT = 300
CAMERA_FOV = 90

ROUTE_DISTANCE_MIN = 80.0
ROUTE_DISTANCE_MAX = 300.0

ROUTE_LOOK_AHEAD = 5
TURN_COMMAND_DISTANCE = 10.0

# ============================================================
# SAFETY SETTINGS
# ============================================================

MAX_THROTTLE = 0.65
MAX_BRAKE = 1.0

MAX_STEER = 1.0

MAX_STEER_CHANGE = 0.12

EMERGENCY_DISTANCE = 6.0

LOW_SPEED = 2.0
MEDIUM_SPEED = 6.0

LOW_SPEED_THROTTLE = 0.40
MEDIUM_SPEED_THROTTLE = 0.55

STUCK_SPEED = 0.5
STUCK_TIME = 5.0


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_image(image):

    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    array = np.reshape(
        array,
        (image.height, image.width, 4)
    )

    rgb_image = array[:, :, :3][:, :, ::-1].copy()

    top = int(0.15 * rgb_image.shape[0])
    bottom = int(0.10 * rgb_image.shape[0])

    rgb_image = rgb_image[top:-bottom, :, :]

    return rgb_image


# ============================================================
# ROAD OPTION -> MODEL COMMAND
# ============================================================

def road_option_to_command(road_option):

    if road_option == RoadOption.LEFT:
        return 1.0

    if road_option == RoadOption.RIGHT:
        return 2.0

    if road_option == RoadOption.STRAIGHT:
        return 3.0

    if road_option == RoadOption.LANEFOLLOW:
        return 4.0

    if road_option == RoadOption.CHANGELANELEFT:
        return 5.0

    if road_option == RoadOption.CHANGELANERIGHT:
        return 6.0

    return 4.0


# ============================================================
# COMMAND NAME
# ============================================================

def command_to_name(command):

    return {
        1.0: "LEFT",
        2.0: "RIGHT",
        3.0: "STRAIGHT",
        4.0: "LANEFOLLOW",
        5.0: "CHANGE_LEFT",
        6.0: "CHANGE_RIGHT"
    }.get(
        command,
        "UNKNOWN"
    )


# ============================================================
# FIND CLOSEST ROUTE INDEX
# ============================================================

def get_closest_route_index(vehicle, route):

    if route is None or len(route) == 0:
        return 0

    vehicle_location = vehicle.get_location()

    closest_index = 0
    closest_distance = float("inf")

    for i, (waypoint, road_option) in enumerate(route):

        distance = vehicle_location.distance(
            waypoint.transform.location
        )

        if distance < closest_distance:

            closest_distance = distance
            closest_index = i

    return closest_index


# ============================================================
# GET CURRENT ROUTE COMMAND
# ============================================================

def get_current_command(vehicle, route):

    if route is None or len(route) == 0:
        return 4.0

    closest_index = get_closest_route_index(
        vehicle,
        route
    )

    closest_waypoint, closest_option = route[closest_index]

    closest_distance = vehicle.get_location().distance(
        closest_waypoint.transform.location
    )

    closest_command = road_option_to_command(
        closest_option
    )

    if closest_command != 4.0:
        return closest_command

    end_index = min(
        closest_index + ROUTE_LOOK_AHEAD,
        len(route)
    )

    for i in range(
        closest_index + 1,
        end_index
    ):

        waypoint, road_option = route[i]

        command = road_option_to_command(
            road_option
        )

        if command != 4.0:

            distance = vehicle.get_location().distance(
                waypoint.transform.location
            )

            if distance <= TURN_COMMAND_DISTANCE:
                return command

            break

    return 4.0


# ============================================================
# GET OBJECT TELEMETRY
# ============================================================

def get_object_telemetry(
    world,
    vehicle,
    max_distance=30.0
):

    vehicle_transform = vehicle.get_transform()

    vehicle_location = (
        vehicle_transform.location
    )

    forward_vector = (
        vehicle_transform.get_forward_vector()
    )

    best_distance = float("inf")
    best_type = 0.0
    best_id = -1.0


    # ========================================================
    # VEHICLES
    # ========================================================

    for actor in world.get_actors().filter(
        "vehicle.*"
    ):

        if actor.id == vehicle.id:
            continue

        location = actor.get_location()

        distance = vehicle_location.distance(
            location
        )

        if distance > max_distance:
            continue

        relative = location - vehicle_location

        forward_dot = (
            relative.x * forward_vector.x
            + relative.y * forward_vector.y
            + relative.z * forward_vector.z
        )

        if forward_dot <= 0:
            continue

        if distance < best_distance:

            best_distance = distance
            best_type = 3.0
            best_id = float(actor.id)


    # ========================================================
    # PEDESTRIANS
    # ========================================================

    for actor in world.get_actors().filter(
        "walker.pedestrian.*"
    ):

        location = actor.get_location()

        distance = vehicle_location.distance(
            location
        )

        if distance > max_distance:
            continue

        relative = location - vehicle_location

        forward_dot = (
            relative.x * forward_vector.x
            + relative.y * forward_vector.y
            + relative.z * forward_vector.z
        )

        if forward_dot <= 0:
            continue

        if distance < best_distance:

            best_distance = distance
            best_type = 4.0
            best_id = float(actor.id)


    # ========================================================
    # TRAFFIC LIGHTS
    # ========================================================

    for actor in world.get_actors().filter(
        "traffic.traffic_light"
    ):

        location = actor.get_location()

        distance = vehicle_location.distance(
            location
        )

        if distance > max_distance:
            continue

        relative = location - vehicle_location

        forward_dot = (
            relative.x * forward_vector.x
            + relative.y * forward_vector.y
            + relative.z * forward_vector.z
        )

        if forward_dot <= 0:
            continue

        if distance < best_distance:

            best_distance = distance
            best_type = 2.0
            best_id = float(actor.id)


    # ========================================================
    # STOP SIGNS
    # ========================================================

    for actor in world.get_actors().filter(
        "traffic.stop"
    ):

        location = actor.get_location()

        distance = vehicle_location.distance(
            location
        )

        if distance > max_distance:
            continue

        relative = location - vehicle_location

        forward_dot = (
            relative.x * forward_vector.x
            + relative.y * forward_vector.y
            + relative.z * forward_vector.z
        )

        if forward_dot <= 0:
            continue

        if distance < best_distance:

            best_distance = distance
            best_type = 1.0
            best_id = float(actor.id)


    # ========================================================
    # NOTHING FOUND
    # ========================================================

    if best_type == 0.0:

        return (
            0.0,
            -1.0,
            0.0
        )

    return (
        best_type,
        best_id,
        float(best_distance)
    )


# ============================================================
# SAFETY CHECK FOR OBJECT IN FRONT
# ============================================================

def get_emergency_distance(
    world,
    vehicle,
    max_distance=EMERGENCY_DISTANCE
):

    vehicle_transform = vehicle.get_transform()

    vehicle_location = (
        vehicle_transform.location
    )

    forward_vector = (
        vehicle_transform.get_forward_vector()
    )

    closest_distance = float("inf")

    actors = world.get_actors()

    for actor in actors:

        if actor.id == vehicle.id:
            continue

        if not (
            actor.type_id.startswith("vehicle.")
            or actor.type_id.startswith("walker.")
        ):
            continue

        location = actor.get_location()

        distance = vehicle_location.distance(
            location
        )

        if distance > max_distance:
            continue

        relative = location - vehicle_location

        forward_dot = (
            relative.x * forward_vector.x
            + relative.y * forward_vector.y
            + relative.z * forward_vector.z
        )

        if forward_dot <= 0:
            continue

        # Approximate lateral distance
        forward_distance = forward_dot

        lateral_distance = abs(
            relative.x * (-forward_vector.y)
            + relative.y * forward_vector.x
        )

        if lateral_distance > 2.5:
            continue

        if forward_distance < closest_distance:

            closest_distance = forward_distance

    return closest_distance


# ============================================================
# RANDOM ROUTE
# ============================================================

def create_random_route(
    world,
    route_planner
):

    spawn_points = (
        world.get_map().get_spawn_points()
    )

    if len(spawn_points) < 2:

        raise RuntimeError(
            "Not enough spawn points."
        )

    attempts = 0

    while attempts < 100:

        attempts += 1

        start = random.choice(
            spawn_points
        )

        destination = random.choice(
            spawn_points
        )

        if start == destination:
            continue

        distance = (
            start.location.distance(
                destination.location
            )
        )

        if not (
            ROUTE_DISTANCE_MIN
            <= distance
            <= ROUTE_DISTANCE_MAX
        ):
            continue

        route = route_planner.trace_route(
            start.location,
            destination.location
        )

        if route is not None and len(route) > 0:

            return (
                start,
                destination,
                route
            )

    raise RuntimeError(
        "Could not find a valid random route."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    vehicle = None
    camera = None

    original_settings = None

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "Loading model..."
    )

    model = AutonomousDriver().to(
        device
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    print(
        "Model loaded successfully."
    )


    # ========================================================
    # IMAGE TRANSFORM
    # ========================================================

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(
            (IMAGE_HEIGHT, IMAGE_WIDTH)
        ),
        transforms.ToTensor()
    ])


    # ========================================================
    # CONNECT TO CARLA
    # ========================================================

    print(
        "Connecting to CARLA..."
    )

    client = carla.Client(
        "localhost",
        2000
    )

    client.set_timeout(
        20.0
    )


    # ========================================================
    # LOAD TOWN02
    # ========================================================

    print(
        "Loading Town02..."
    )

    world = client.load_world(
        "Town02"
    )

    print(
        "Town02 loaded."
    )


    # ========================================================
    # SYNCHRONOUS MODE
    # ========================================================

    original_settings = (
        world.get_settings()
    )

    settings = (
        world.get_settings()
    )

    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05

    world.apply_settings(
        settings
    )


    # ========================================================
    # BLUEPRINTS
    # ========================================================

    blueprint_library = (
        world.get_blueprint_library()
    )


    # ========================================================
    # VEHICLE
    # ========================================================

    vehicle_blueprints = (
        blueprint_library.filter(
            "vehicle.tesla.model3"
        )
    )

    if len(vehicle_blueprints) == 0:

        raise RuntimeError(
            "Tesla Model 3 blueprint was not found."
        )

    vehicle_bp = (
        vehicle_blueprints[0]
    )


    # ========================================================
    # ROUTE PLANNER
    # ========================================================

    print(
        "Creating route planner..."
    )

    route_planner = GlobalRoutePlanner(
        world.get_map(),
        2.0
    )


    # ========================================================
    # RANDOM ROUTE
    # ========================================================

    print(
        "Finding random route..."
    )

    (
        spawn_point,
        destination,
        route
    ) = create_random_route(
        world,
        route_planner
    )

    print(
        f"Route generated: "
        f"{len(route)} waypoints"
    )

    print(
        f"Start: "
        f"{spawn_point.location.x:.1f}, "
        f"{spawn_point.location.y:.1f}"
    )

    print(
        f"Destination: "
        f"{destination.location.x:.1f}, "
        f"{destination.location.y:.1f}"
    )


    # ========================================================
    # SPAWN VEHICLE
    # ========================================================

    spawn_transform = carla.Transform(
        carla.Location(
            x=spawn_point.location.x,
            y=spawn_point.location.y,
            z=spawn_point.location.z + 0.2
        ),
        spawn_point.rotation
    )

    vehicle = world.spawn_actor(
        vehicle_bp,
        spawn_transform
    )

    if vehicle is None:

        raise RuntimeError(
            "Vehicle failed to spawn."
        )

    print(
        "Vehicle spawned."
    )


    # ========================================================
    # CAMERA
    # ========================================================

    camera_bp = (
        blueprint_library.find(
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


    camera_transform = carla.Transform(
        carla.Location(
            x=1.3,
            y=0.0,
            z=2.0
        ),
        carla.Rotation(
            pitch=-5.0,
            yaw=0.0,
            roll=0.0
        )
    )


    camera = world.spawn_actor(
        camera_bp,
        camera_transform,
        attach_to=vehicle
    )


    # ========================================================
    # CAMERA QUEUE
    # ========================================================

    image_queue = queue.Queue(
        maxsize=2
    )

    camera.listen(
        image_queue.put
    )


    # ========================================================
    # SPECTATOR
    # ========================================================

    spectator = (
        world.get_spectator()
    )


    # ========================================================
    # PREVIOUS CONTROL BRAKE
    # ========================================================

    previous_control_brake = 0.0


    # ========================================================
    # PREVIOUS STEERING
    # ========================================================

    previous_steer = 0.0


    # ========================================================
    # STUCK TIMER
    # ========================================================

    stuck_time = 0.0


    # ========================================================
    # START
    # ========================================================

    print(
        "\nStarting autonomous driving...\n"
    )


    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while True:

            # =================================================
            # SIMULATION TICK
            # =================================================

            frame = world.tick()


            # =================================================
            # GET CAMERA IMAGE
            # =================================================

            image_data = None

            while True:

                candidate = (
                    image_queue.get()
                )

                if candidate.frame == frame:

                    image_data = candidate

                    break


            # =================================================
            # PROCESS IMAGE
            # =================================================

            latest_image = process_image(
                image_data
            )


            # =================================================
            # DISPLAY CAMERA
            # =================================================

            display_image = cv2.cvtColor(
                latest_image,
                cv2.COLOR_RGB2BGR
            )

            cv2.imshow(
                "AI Camera Feed",
                display_image
            )

            key = (
                cv2.waitKey(1) & 0xFF
            )

            if key == ord("q"):

                print(
                    "\nQuit requested."
                )

                break


            # =================================================
            # SPEED
            # =================================================

            velocity = (
                vehicle.get_velocity()
            )

            speed = math.sqrt(
                velocity.x ** 2
                + velocity.y ** 2
                + velocity.z ** 2
            )


            # =================================================
            # ROUTE COMMAND
            # =================================================

            command = get_current_command(
                vehicle,
                route
            )


            # =================================================
            # JUNCTION
            # =================================================

            vehicle_waypoint = (
                world.get_map().get_waypoint(
                    vehicle.get_location(),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )
            )

            if vehicle_waypoint is not None:

                junction = float(
                    vehicle_waypoint.is_junction
                )

            else:

                junction = 0.0


            # =================================================
            # OBJECT TELEMETRY
            # =================================================

            (
                speed_reduced_by_obj_type,
                speed_reduced_by_obj_id,
                speed_reduced_by_obj_distance
            ) = get_object_telemetry(
                world,
                vehicle
            )


            # =================================================
            # SPEED LIMIT
            # =================================================

            speed_limit = float(
                vehicle.get_speed_limit()
            )


            # =================================================
            # TELEMETRY
            #
            # 1. speed
            # 2. command
            # 3. junction
            # 4. object type
            # 5. object id
            # 6. object distance
            # 7. speed limit
            # 8. control brake
            # =================================================

            telemetry_tensor = torch.tensor(
                [[
                    speed,
                    command,
                    junction,
                    speed_reduced_by_obj_type,
                    speed_reduced_by_obj_id,
                    speed_reduced_by_obj_distance,
                    speed_limit,
                    previous_control_brake
                ]],
                dtype=torch.float32,
                device=device
            )


            # =================================================
            # IMAGE TENSOR
            # =================================================

            image_tensor = transform(
                latest_image
            ).unsqueeze(0).to(
                device
            )


            # =================================================
            # MODEL INFERENCE
            # =================================================

            with torch.no_grad():

                output = model(
                    image_tensor,
                    telemetry_tensor
                )

                output = output[0]

                raw_steer = output[0].item()
                raw_throttle = output[1].item()
                raw_brake = output[2].item()


            # =================================================
            # SANITIZE NETWORK OUTPUT
            # =================================================

            if not math.isfinite(raw_steer):
                raw_steer = 0.0

            if not math.isfinite(raw_throttle):
                raw_throttle = 0.0

            if not math.isfinite(raw_brake):
                raw_brake = 0.0


            # =================================================
            # CLAMP RAW OUTPUTS
            # =================================================

            steer = max(
                -MAX_STEER,
                min(
                    MAX_STEER,
                    raw_steer
                )
            )

            throttle = max(
                0.0,
                min(
                    MAX_THROTTLE,
                    raw_throttle
                )
            )

            brake = max(
                0.0,
                min(
                    MAX_BRAKE,
                    raw_brake
                )
            )


            # =================================================
            # STEERING RATE LIMIT
            #
            # Prevents a single bad frame from changing
            # steering violently.
            # =================================================

            steer_difference = (
                steer - previous_steer
            )

            if steer_difference > MAX_STEER_CHANGE:

                steer = (
                    previous_steer
                    + MAX_STEER_CHANGE
                )

            elif steer_difference < -MAX_STEER_CHANGE:

                steer = (
                    previous_steer
                    - MAX_STEER_CHANGE
                )


            # =================================================
            # SPEED-BASED THROTTLE SAFETY
            # =================================================

            if speed < LOW_SPEED:

                throttle = min(
                    throttle,
                    LOW_SPEED_THROTTLE
                )

            elif speed < MEDIUM_SPEED:

                throttle = min(
                    throttle,
                    MEDIUM_SPEED_THROTTLE
                )


            # =================================================
            # SLOW DOWN FOR HARD TURNS
            # =================================================

            if command in (
                1.0,
                2.0,
                5.0,
                6.0
            ):

                throttle = min(
                    throttle,
                    0.45
                )


            # =================================================
            # EMERGENCY OBJECT CHECK
            # =================================================

            emergency_distance = (
                get_emergency_distance(
                    world,
                    vehicle
                )
            )

            if emergency_distance < 4.0:

                throttle = 0.0
                brake = 1.0

            elif emergency_distance < 6.0:

                throttle = 0.0
                brake = max(
                    brake,
                    0.60
                )


            # =================================================
            # BRAKE / THROTTLE ARBITRATION
            # =================================================

            # Do not allow the learned controller to command
            # meaningful throttle and brake at the same time.
            if brake > 0.10:

                throttle = 0.0

            # If the vehicle is essentially stationary and the
            # model is not issuing a strong emergency brake,
            # allow it to recover from a false low-speed brake.
            if speed < 0.50 and brake < 0.90:

                brake = 0.0
                throttle = max(
                    throttle,
                    0.30
                )


            # =================================================
            # BUILD CARLA CONTROL
            # =================================================

            control = carla.VehicleControl()

            control.steer = float(
                steer
            )

            control.throttle = float(
                throttle
            )

            control.brake = float(
                brake
            )

            control.hand_brake = False
            control.reverse = False


            # =================================================
            # APPLY CONTROL
            # =================================================

            vehicle.apply_control(
                control
            )


            # =================================================
            # UPDATE PREVIOUS VALUES
            # =================================================

            previous_control_brake = (
                control.brake
            )

            previous_steer = (
                control.steer
            )


            # =================================================
            # SPECTATOR CAMERA
            # =================================================

            vehicle_transform = (
                vehicle.get_transform()
            )

            yaw = math.radians(
                vehicle_transform.rotation.yaw
            )

            spectator_location = carla.Location(
                x=(
                    vehicle_transform.location.x
                    - 8.0 * math.cos(yaw)
                ),
                y=(
                    vehicle_transform.location.y
                    - 8.0 * math.sin(yaw)
                ),
                z=(
                    vehicle_transform.location.z
                    + 3.0
                )
            )

            spectator.set_transform(
                carla.Transform(
                    spectator_location,
                    vehicle_transform.rotation
                )
            )


            # =================================================
            # COMMAND NAME
            # =================================================

            command_name = command_to_name(
                command
            )


            # =================================================
            # STATUS
            # =================================================

            print(
                f"\r"
                f"Speed: {speed:.2f} m/s | "
                f"Command: {command_name:<12} | "
                f"Raw S: {raw_steer:+.3f} | "
                f"Raw T: {raw_throttle:+.3f} | "
                f"Raw B: {raw_brake:+.3f} | "
                f"Applied S: {steer:+.3f} | "
                f"T: {throttle:+.3f} | "
                f"B: {brake:+.3f}",
                end="",
                flush=True
            )


    except KeyboardInterrupt:

        print(
            "\n\nStopping..."
        )


    finally:

        print(
            "\n\nCleaning up..."
        )

        cv2.destroyAllWindows()

        if camera is not None:

            camera.stop()
            camera.destroy()

        if vehicle is not None:

            vehicle.destroy()

        if original_settings is not None:

            world.apply_settings(
                original_settings
            )

        print(
            "Cleanup complete."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()