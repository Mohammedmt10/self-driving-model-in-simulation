import carla
import time
import math
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import queue

from model import AutonomousDriver


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "carla_driver_pytorch_v4_best.pth"

# These MUST match the input size used during training
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200

# CARLA camera resolution
CAMERA_WIDTH = 400
CAMERA_HEIGHT = 300
CAMERA_FOV = 90


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_image(image):
    """
    Convert CARLA BGRA image -> RGB image.

    Applies the same crop used during training:
        Remove top 30%
        Remove bottom 10%

    Returns an RGB numpy array.
    """

    # CARLA image is BGRA
    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8
    )

    array = np.reshape(
        array,
        (image.height, image.width, 4)
    )

    # BGRA -> RGB
    rgb_image = array[:, :, :3][:, :, ::-1].copy()

    # --------------------------------------------------------
    # IMPORTANT:
    # This must EXACTLY match dataset.py
    # --------------------------------------------------------

    height = rgb_image.shape[0]

    top = int(0.30 * height)
    bottom = int(0.10 * height)

    if bottom > 0:
        cropped_image = rgb_image[top:-bottom, :, :]
    else:
        cropped_image = rgb_image[top:, :, :]

    return cropped_image


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. DEVICE
    # ========================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")


    # ========================================================
    # 2. LOAD MODEL
    # ========================================================

    print("Loading model...")

    model = AutonomousDriver().to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.eval()

    print("Model loaded successfully.")


    # ========================================================
    # 3. IMAGE TRANSFORM
    # ========================================================

    # MUST match dataset.py
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
    ])


    # ========================================================
    # 4. CONNECT TO CARLA
    # ========================================================

    print("Connecting to CARLA...")

    client = carla.Client(
        "localhost",
        2000
    )

    client.set_timeout(20.0)


    # ========================================================
    # 5. LOAD TOWN
    # ========================================================

    print("Loading Town01...")

    world = client.load_world("Town01")

    print("Town01 loaded.")


    # ========================================================
    # 6. SYNCHRONOUS MODE
    # ========================================================

    original_settings = world.get_settings()

    settings = world.get_settings()

    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05

    world.apply_settings(settings)

    print("Synchronous mode enabled.")


    # ========================================================
    # 7. BLUEPRINTS
    # ========================================================

    blueprint_library = world.get_blueprint_library()


    # ========================================================
    # 8. SPAWN VEHICLE
    # ========================================================

    vehicle_blueprints = blueprint_library.filter(
        "vehicle.tesla.model3"
    )

    if len(vehicle_blueprints) == 0:
        raise RuntimeError(
            "Tesla Model 3 blueprint was not found."
        )

    vehicle_bp = vehicle_blueprints[0]

    spawn_points = world.get_map().get_spawn_points()

    if len(spawn_points) == 0:
        raise RuntimeError(
            "No spawn points available in Town01."
        )

    spawn_point = spawn_points[0]

    # Slightly raise vehicle to prevent spawning inside road
    spawn_point.location.z += 0.2

    vehicle = world.spawn_actor(
        vehicle_bp,
        spawn_point
    )

    print("Vehicle spawned.")


    # ========================================================
    # 9. CAMERA
    # ========================================================

    camera = None

    camera_bp = blueprint_library.find(
        "sensor.camera.rgb"
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

    # --------------------------------------------------------
    # IMPORTANT CAMERA POSITION
    #
    # This is the AI's POV.
    #
    # x = forward/backward
    # y = left/right
    # z = height
    #
    # The image you provided is a centered forward hood view.
    # --------------------------------------------------------

    camera_transform = carla.Transform(
        carla.Location(
            x=1.5,       # slightly in front of windshield
            y=0.0,       # perfectly centered
            z=1.4        # cabin/dashboard height
        ),
        carla.Rotation(
            pitch=0.0,   # looking straight ahead
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
    # 10. IMAGE QUEUE
    # ========================================================

    image_queue = queue.Queue(maxsize=2)

    camera.listen(
        image_queue.put
    )

    print("Camera active.")
    print("Starting autonomous driving...\n")


    # ========================================================
    # 11. SPECTATOR CAMERA
    # ========================================================

    spectator = world.get_spectator()


    # ========================================================
    # 12. MAIN LOOP
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # Advance exactly one simulation frame
            # ------------------------------------------------

            frame = world.tick()


            # ------------------------------------------------
            # Get camera image corresponding to this frame
            # ------------------------------------------------

            image_data = None

            while True:

                candidate = image_queue.get()

                if candidate.frame == frame:
                    image_data = candidate
                    break

                # Ignore old frames


            # ------------------------------------------------
            # Process image
            # ------------------------------------------------

            latest_image = process_image(
                image_data
            )


            # ------------------------------------------------
            # DISPLAY EXACTLY WHAT AI SEES
            # ------------------------------------------------

            display_image = cv2.cvtColor(
                latest_image,
                cv2.COLOR_RGB2BGR
            )

            cv2.imshow(
                "AI Camera Feed - Model Input",
                display_image
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\nQuit requested.")
                break


            # =================================================
            # SPECTATOR CHASE CAMERA
            #
            # This DOES NOT affect the AI camera.
            # It is only for you to watch the vehicle.
            # =================================================

            v_transform = vehicle.get_transform()

            yaw = math.radians(
                v_transform.rotation.yaw
            )

            spectator_x = (
                v_transform.location.x
                - 8.0 * math.cos(yaw)
            )

            spectator_y = (
                v_transform.location.y
                - 8.0 * math.sin(yaw)
            )

            spectator_z = (
                v_transform.location.z
                + 3.0
            )

            spectator.set_transform(
                carla.Transform(
                    carla.Location(
                        x=spectator_x,
                        y=spectator_y,
                        z=spectator_z
                    ),
                    v_transform.rotation
                )
            )


            # =================================================
            # VEHICLE TELEMETRY
            # =================================================

            velocity = vehicle.get_velocity()

            speed = math.sqrt(
                velocity.x ** 2 +
                velocity.y ** 2 +
                velocity.z ** 2
            )


            # ------------------------------------------------
            # Navigation command
            #
            # IMPORTANT:
            # This value must match how your dataset was
            # generated.
            # ------------------------------------------------

            command = 4.0

            telemetry_tensor = torch.tensor(
                [[speed, command]],
                dtype=torch.float32,
                device=device
            )


            # =================================================
            # MODEL INPUT
            # =================================================

            img_tensor = transform(
                latest_image
            ).unsqueeze(0).to(device)


            # =================================================
            # MODEL INFERENCE
            # =================================================

            with torch.no_grad():

                predictions = model(
                    img_tensor,
                    telemetry_tensor
                )

                predictions = predictions[0]

                steer = predictions[0].item()
                throttle = predictions[1].item()
                control_brake = predictions[2].item()


            # =================================================
            # SAFETY / RANGE LIMITS
            # =================================================

            steer = float(
                np.clip(
                    steer,
                    -1.0,
                    1.0
                )
            )

            throttle = float(
                np.clip(
                    throttle,
                    0.0,
                    1.0
                )
            )

            control_brake = float(
                np.clip(
                    control_brake,
                    0.0,
                    1.0
                )
            )


            # Remove extremely small steering noise

            if abs(steer) < 0.01:
                steer = 0.0


            # =================================================
            # APPLY CONTROL
            # =================================================

            control = carla.VehicleControl()

            control.steer = steer
            control.throttle = throttle
            control.brake = control_brake

            control.hand_brake = False
            control.reverse = False

            vehicle.apply_control(control)


            # =================================================
            # TERMINAL STATUS
            # =================================================

            print(
                f"\r"
                f"[Speed: {speed:.2f} m/s] "
                f"Steer: {steer:+.3f} | "
                f"Throttle: {throttle:.3f} | "
                f"Brake: {control_brake:.3f}",
                end="",
                flush=True
            )


    except KeyboardInterrupt:

        print(
            "\n\nStopping inference..."
        )


    finally:

        print(
            "\n\nCleaning up..."
        )

        cv2.destroyAllWindows()

        # Restore CARLA settings
        world.apply_settings(
            original_settings
        )

        # Destroy camera
        if camera is not None:
            camera.stop()
            camera.destroy()

        # Destroy vehicle
        if vehicle is not None:
            vehicle.destroy()

        print("Cleanup complete.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()