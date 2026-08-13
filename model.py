import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler
)

from dataset import (
    CarlaDataset,
    build_path_index,
    get_steering_sampling_weights,
    summarize_steering_distribution
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_TELEMETRY = 11

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 2e-5
VAL_RATIO = 0.15
SEED = 42

CHECKPOINT = "model_v8.pth"


# ============================================================
# MODEL
# ============================================================

class AutonomousDriver(nn.Module):

    def __init__(self):

        super().__init__()

        # ====================================================
        # VISION BRANCH
        # Input:
        #   [B, 3, 200, 400]
        #
        # Output:
        #   [B, 500]
        # ====================================================

        self.vision_branch = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                64,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                128,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(
                (10, 10)
            ),

            nn.Flatten(),

            nn.Linear(
                128 * 10 * 10,
                256
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                256,
                500
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                0.2
            )
        )

        # ====================================================
        # VEHICLE STATE
        #
        # Telemetry:
        #
        # 0 speed
        # 6 speed_limit
        # 7 angle
        # 9 theta
        # 10 lateral_distance
        #
        # Total = 5
        #
        # Output = 64
        # ====================================================

        self.vehicle_state_branch = nn.Sequential(

            nn.Linear(
                5,
                32
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                32,
                64
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                64,
                64
            ),

            nn.ReLU(inplace=True)
        )

        # ====================================================
        # JUNCTION
        #
        # Telemetry:
        #
        # 2 junction
        #
        # Output = 32
        # ====================================================

        self.junction_state_branch = nn.Sequential(

            nn.Linear(
                1,
                16
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                16,
                32
            ),

            nn.ReLU(inplace=True)
        )

        # ====================================================
        # COMMAND
        #
        # Values:
        #
        # 1 LEFT
        # 2 RIGHT
        # 3 STRAIGHT
        # 4 LANEFOLLOW
        # 5 CHANGE_LANE_LEFT
        # 6 CHANGE_LANE_RIGHT
        #
        # 0 is reserved.
        # ====================================================

        self.command_embedding = nn.Embedding(
            7,
            32
        )

        # ====================================================
        # NEXT COMMAND
        # ====================================================

        self.next_command_embedding = nn.Embedding(
            7,
            32
        )

        # ====================================================
        # OBJECT TYPE
        #
        # 0 None
        # 1 stop
        # 2 traffic light
        # 3 vehicle
        # 4 pedestrian
        # ====================================================

        self.object_type_embedding = nn.Embedding(
            5,
            16
        )

        # ====================================================
        # OBJECT STATE
        #
        # 16 object-type embedding
        # 1 object ID
        # 1 object distance
        #
        # Total = 18
        #
        # Output = 64
        # ====================================================

        self.object_state_branch = nn.Sequential(

            nn.Linear(
                18,
                32
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                32,
                64
            ),

            nn.ReLU(inplace=True)
        )

        # ====================================================
        # FUSION
        #
        # Vehicle       = 64
        # Junction      = 32
        # Command       = 32
        # Next command  = 32
        # Object        = 64
        # Vision        = 500
        #
        # TOTAL         = 724
        # ====================================================

        self.fusion_branch = nn.Sequential(

            nn.Linear(
                724,
                256
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(inplace=True)
        )

        # ====================================================
        # STEERING HEAD
        #
        # Output range:
        # [-1, 1]
        # ====================================================

        self.steering_head = nn.Sequential(

            nn.Linear(
                128,
                1
            ),

            nn.Tanh()
        )

        # ====================================================
        # THROTTLE HEAD
        #
        # Output range:
        # [0, 1]
        # ====================================================

        self.throttle_head = nn.Sequential(

            nn.Linear(
                128,
                1
            ),

            nn.Sigmoid()
        )

        # ====================================================
        # BRAKE HEAD
        #
        # IMPORTANT:
        #
        # This outputs a LOGIT.
        #
        # sigmoid() is applied only when interpreting
        # the output as a probability.
        # ====================================================

        self.brake_head = nn.Linear(
            128,
            1
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        image_input,
        telemetry_input
    ):

        # ====================================================
        # CHECK TELEMETRY
        # ====================================================

        if telemetry_input.ndim != 2:

            raise RuntimeError(
                "Telemetry must have shape [B, 11]. "
                f"Got {telemetry_input.shape}"
            )

        if telemetry_input.shape[1] != NUM_TELEMETRY:

            raise RuntimeError(
                "Model expects exactly "
                f"{NUM_TELEMETRY} telemetry values. "
                f"Got {telemetry_input.shape[1]}"
            )

        # ====================================================
        # TELEMETRY INDICES
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

        speed = telemetry_input[:, 0:1]

        command = (
            telemetry_input[:, 1]
            .long()
            .clamp(0, 6)
        )

        junction = telemetry_input[:, 2:3]

        object_type = (
            telemetry_input[:, 3]
            .long()
            .clamp(0, 4)
        )

        object_id = telemetry_input[:, 4:5]

        object_distance = telemetry_input[:, 5:6]

        speed_limit = telemetry_input[:, 6:7]

        angle = telemetry_input[:, 7:8]

        next_command = (
            telemetry_input[:, 8]
            .long()
            .clamp(0, 6)
        )

        theta = telemetry_input[:, 9:10]

        lateral_distance = telemetry_input[:, 10:11]

        # ====================================================
        # VEHICLE STATE
        # ====================================================

        vehicle_state = torch.cat(
            (
                speed,
                speed_limit,
                angle,
                theta,
                lateral_distance
            ),
            dim=1
        )

        vehicle_features = (
            self.vehicle_state_branch(
                vehicle_state
            )
        )

        # ====================================================
        # JUNCTION
        # ====================================================

        junction_features = (
            self.junction_state_branch(
                junction
            )
        )

        # ====================================================
        # COMMAND
        # ====================================================

        command_features = (
            self.command_embedding(
                command
            )
        )

        # ====================================================
        # NEXT COMMAND
        # ====================================================

        next_command_features = (
            self.next_command_embedding(
                next_command
            )
        )

        # ====================================================
        # OBJECT TYPE
        # ====================================================

        object_type_features = (
            self.object_type_embedding(
                object_type
            )
        )

        # ====================================================
        # OBJECT STATE
        # ====================================================

        object_state = torch.cat(
            (
                object_type_features,
                object_id,
                object_distance
            ),
            dim=1
        )

        object_features = (
            self.object_state_branch(
                object_state
            )
        )

        # ====================================================
        # IMAGE
        # ====================================================

        vision_features = (
            self.vision_branch(
                image_input
            )
        )

        # ====================================================
        # FUSION
        # ====================================================

        combined_features = torch.cat(
            (
                vehicle_features,
                junction_features,
                command_features,
                next_command_features,
                object_features,
                vision_features
            ),
            dim=1
        )

        if combined_features.shape[1] != 724:

            raise RuntimeError(
                "Fusion feature size mismatch. "
                f"Expected 724, "
                f"got {combined_features.shape[1]}"
            )

        fused_features = (
            self.fusion_branch(
                combined_features
            )
        )

        # ====================================================
        # OUTPUTS
        # ====================================================

        steering = (
            self.steering_head(
                fused_features
            )
        )

        throttle = (
            self.throttle_head(
                fused_features
            )
        )

        brake_logit = (
            self.brake_head(
                fused_features
            )
        )

        # ====================================================
        # FINAL OUTPUT
        #
        # [steering, throttle, brake_logit]
        # ====================================================

        return torch.cat(
            (
                steering,
                throttle,
                brake_logit
            ),
            dim=1
        )


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=15,
    learning_rate=2e-5
):

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    # ========================================================
    # LOSSES
    # ========================================================

    regression_criterion = nn.HuberLoss(
        reduction="none"
    )

    brake_criterion = nn.BCEWithLogitsLoss(
        reduction="none"
    )

    # ========================================================
    # OUTPUT WEIGHTS
    #
    # Steering receives 2x weight.
    # ========================================================

    output_weights = torch.tensor(
        [
            2.0,
            1.0,
            1.0
        ],
        dtype=torch.float32,
        device=device
    )

    best_val_loss = float("inf")
    best_epoch = 0

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(epochs):

        # ====================================================
        # TRAIN MODE
        # ====================================================

        model.train()

        running_loss = 0.0

        train_absolute_error = torch.zeros(
            3,
            device=device
        )

        train_samples = 0

        # ====================================================
        # TRAIN BATCHES
        # ====================================================

        for batch_idx, (
            images,
            telemetry,
            targets
        ) in enumerate(
            train_loader,
            start=1
        ):

            images = images.to(
                device,
                non_blocking=True
            )

            telemetry = telemetry.to(
                device,
                non_blocking=True
            )

            targets = targets.to(
                device,
                non_blocking=True
            )

            # ------------------------------------------------
            # VALIDATION OF INPUT SHAPES
            # ------------------------------------------------

            if telemetry.shape[1] != NUM_TELEMETRY:

                raise RuntimeError(
                    "Training dataset returned "
                    f"{telemetry.shape[1]} telemetry values. "
                    f"Expected {NUM_TELEMETRY}."
                )

            if targets.shape[1] != 3:

                raise RuntimeError(
                    "Training dataset returned "
                    f"{targets.shape[1]} targets. "
                    "Expected 3."
                )

            # ------------------------------------------------
            # NUMERICAL CHECK
            # ------------------------------------------------

            if not torch.isfinite(images).all():

                raise RuntimeError(
                    "Image contains NaN/Inf."
                )

            if not torch.isfinite(telemetry).all():

                raise RuntimeError(
                    "Telemetry contains NaN/Inf."
                )

            if not torch.isfinite(targets).all():

                raise RuntimeError(
                    "Targets contain NaN/Inf."
                )

            # ------------------------------------------------
            # ZERO GRADIENT
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # FORWARD
            # ------------------------------------------------

            outputs = model(
                images,
                telemetry
            )

            # =================================================
            # STEERING LOSS
            # =================================================

            steer_loss = regression_criterion(
                outputs[:, 0],
                targets[:, 0]
            )

            # =================================================
            # THROTTLE LOSS
            # =================================================

            throttle_loss = regression_criterion(
                outputs[:, 1],
                targets[:, 1]
            )

            # =================================================
            # BRAKE LOSS
            # =================================================

            brake_loss = brake_criterion(
                outputs[:, 2],
                targets[:, 2]
            )

            # =================================================
            # COMBINE
            # =================================================

            individual_loss = torch.stack(
                (
                    steer_loss,
                    throttle_loss,
                    brake_loss
                ),
                dim=1
            )

            weighted_loss = (
                individual_loss
                * output_weights
            )

            loss = weighted_loss.mean()

            # =================================================
            # BACKPROP
            # =================================================

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            # =================================================
            # METRICS
            # =================================================

            batch_size = images.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            with torch.no_grad():

                brake_probability = torch.sigmoid(
                    outputs[:, 2]
                )

                brake_prediction = (
                    brake_probability >= 0.5
                ).float()

                batch_error = torch.stack(
                    (
                        torch.abs(
                            outputs[:, 0]
                            -
                            targets[:, 0]
                        ),

                        torch.abs(
                            outputs[:, 1]
                            -
                            targets[:, 1]
                        ),

                        torch.abs(
                            brake_prediction
                            -
                            targets[:, 2]
                        )
                    ),
                    dim=1
                )

                train_absolute_error += (
                    batch_error.sum(
                        dim=0
                    )
                )

            train_samples += batch_size

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if batch_idx % 100 == 0:

                print(
                    f"[Epoch {epoch + 1}, "
                    f"Batch {batch_idx}] "
                    f"train_loss: "
                    f"{running_loss / train_samples:.4f}"
                )

        # ====================================================
        # TRAIN METRICS
        # ====================================================

        train_loss = (
            running_loss
            /
            max(train_samples, 1)
        )

        train_mae = (
            train_absolute_error
            /
            max(train_samples, 1)
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss_total = 0.0

        val_output_loss = torch.zeros(
            3,
            device=device
        )

        val_absolute_error = torch.zeros(
            3,
            device=device
        )

        val_samples = 0

        # ====================================================
        # BRAKE CONFUSION MATRIX
        # ====================================================

        brake_tp = 0
        brake_tn = 0
        brake_fp = 0
        brake_fn = 0

        # ====================================================
        # STEERING METRICS
        # ====================================================

        steer_sign_correct = 0
        steer_sign_total = 0

        strong_steer_error = 0.0
        strong_steer_samples = 0

        # ====================================================
        # VALIDATION LOOP
        # ====================================================

        with torch.no_grad():

            for (
                images,
                telemetry,
                targets
            ) in val_loader:

                images = images.to(
                    device,
                    non_blocking=True
                )

                telemetry = telemetry.to(
                    device,
                    non_blocking=True
                )

                targets = targets.to(
                    device,
                    non_blocking=True
                )

                # ------------------------------------------------
                # SHAPE CHECK
                # ------------------------------------------------

                if telemetry.shape[1] != NUM_TELEMETRY:

                    raise RuntimeError(
                        "Validation dataset returned "
                        f"{telemetry.shape[1]} telemetry values. "
                        f"Expected {NUM_TELEMETRY}."
                    )

                # ------------------------------------------------
                # FORWARD
                # ------------------------------------------------

                outputs = model(
                    images,
                    telemetry
                )

                # =================================================
                # LOSSES
                # =================================================

                steer_loss = regression_criterion(
                    outputs[:, 0],
                    targets[:, 0]
                )

                throttle_loss = regression_criterion(
                    outputs[:, 1],
                    targets[:, 1]
                )

                brake_loss = brake_criterion(
                    outputs[:, 2],
                    targets[:, 2]
                )

                individual_loss = torch.stack(
                    (
                        steer_loss,
                        throttle_loss,
                        brake_loss
                    ),
                    dim=1
                )

                weighted_loss = (
                    individual_loss
                    * output_weights
                )

                loss = weighted_loss.mean()

                batch_size = images.size(0)

                val_loss_total += (
                    loss.item()
                    * batch_size
                )

                val_output_loss += (
                    individual_loss.sum(
                        dim=0
                    )
                )

                # =================================================
                # BRAKE PREDICTION
                # =================================================

                brake_probability = torch.sigmoid(
                    outputs[:, 2]
                )

                brake_prediction = (
                    brake_probability >= 0.5
                ).float()

                # =================================================
                # MAE
                # =================================================

                batch_error = torch.stack(
                    (
                        torch.abs(
                            outputs[:, 0]
                            -
                            targets[:, 0]
                        ),

                        torch.abs(
                            outputs[:, 1]
                            -
                            targets[:, 1]
                        ),

                        torch.abs(
                            brake_prediction
                            -
                            targets[:, 2]
                        )
                    ),
                    dim=1
                )

                val_absolute_error += (
                    batch_error.sum(
                        dim=0
                    )
                )

                # =================================================
                # BRAKE CONFUSION MATRIX
                # =================================================

                actual_brake = targets[:, 2]

                brake_tp += (
                    (
                        (brake_prediction == 1)
                        &
                        (actual_brake == 1)
                    )
                    .sum()
                    .item()
                )

                brake_tn += (
                    (
                        (brake_prediction == 0)
                        &
                        (actual_brake == 0)
                    )
                    .sum()
                    .item()
                )

                brake_fp += (
                    (
                        (brake_prediction == 1)
                        &
                        (actual_brake == 0)
                    )
                    .sum()
                    .item()
                )

                brake_fn += (
                    (
                        (brake_prediction == 0)
                        &
                        (actual_brake == 1)
                    )
                    .sum()
                    .item()
                )

                # =================================================
                # STEERING SIGN ACCURACY
                # =================================================

                meaningful = (
                    torch.abs(
                        targets[:, 0]
                    )
                    >=
                    0.05
                )

                if meaningful.any():

                    true_sign = torch.sign(
                        targets[:, 0][meaningful]
                    )

                    pred_sign = torch.sign(
                        outputs[:, 0][meaningful]
                    )

                    steer_sign_correct += (
                        (
                            true_sign
                            ==
                            pred_sign
                        )
                        .sum()
                        .item()
                    )

                    steer_sign_total += (
                        meaningful.sum().item()
                    )

                # =================================================
                # STRONG TURN MAE
                # =================================================

                strong = (
                    torch.abs(
                        targets[:, 0]
                    )
                    >=
                    0.25
                )

                if strong.any():

                    strong_error = torch.abs(
                        outputs[:, 0][strong]
                        -
                        targets[:, 0][strong]
                    )

                    strong_steer_error += (
                        strong_error.sum().item()
                    )

                    strong_steer_samples += (
                        strong.sum().item()
                    )

                val_samples += batch_size

        # ====================================================
        # VALIDATION METRICS
        # ====================================================

        val_loss = (
            val_loss_total
            /
            max(val_samples, 1)
        )

        val_mae = (
            val_absolute_error
            /
            max(val_samples, 1)
        )

        val_output_loss_avg = (
            val_output_loss
            /
            max(val_samples, 1)
        )

        # ====================================================
        # BRAKE METRICS
        # ====================================================

        brake_total = (
            brake_tp
            +
            brake_tn
            +
            brake_fp
            +
            brake_fn
        )

        brake_accuracy = (
            (brake_tp + brake_tn)
            /
            brake_total
            if brake_total > 0
            else 0.0
        )

        brake_precision = (
            brake_tp
            /
            (brake_tp + brake_fp)
            if (brake_tp + brake_fp) > 0
            else 0.0
        )

        brake_recall = (
            brake_tp
            /
            (brake_tp + brake_fn)
            if (brake_tp + brake_fn) > 0
            else 0.0
        )

        # ====================================================
        # STEERING METRICS
        # ====================================================

        steer_sign_accuracy = (
            steer_sign_correct
            /
            steer_sign_total
            if steer_sign_total > 0
            else 0.0
        )

        strong_turn_mae = (
            strong_steer_error
            /
            strong_steer_samples
            if strong_steer_samples > 0
            else 0.0
        )

        # ====================================================
        # PRINT
        # ====================================================

        print()
        print("=" * 80)

        print(
            f"--- End of Epoch {epoch + 1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} ---"
        )

        print(
            f"Train MAE | "
            f"Steer: {train_mae[0]:.4f} | "
            f"Throttle: {train_mae[1]:.4f} | "
            f"Brake: {train_mae[2]:.4f}"
        )

        print(
            f"Val MAE   | "
            f"Steer: {val_mae[0]:.4f} | "
            f"Throttle: {val_mae[1]:.4f} | "
            f"Brake: {val_mae[2]:.4f}"
        )

        print(
            f"Val Loss  | "
            f"Steer: {val_output_loss_avg[0]:.4f} | "
            f"Throttle: {val_output_loss_avg[1]:.4f} | "
            f"Brake BCE: {val_output_loss_avg[2]:.4f}"
        )

        print(
            f"Steer Metrics | "
            f"Sign Accuracy: "
            f"{steer_sign_accuracy:.4f} | "
            f"Strong-turn MAE: "
            f"{strong_turn_mae:.4f}"
        )

        print(
            f"Brake Metrics | "
            f"Accuracy: "
            f"{brake_accuracy:.4f} | "
            f"Precision: "
            f"{brake_precision:.4f} | "
            f"Recall: "
            f"{brake_recall:.4f}"
        )

        print(
            f"Brake Confusion | "
            f"TP: {brake_tp} | "
            f"TN: {brake_tn} | "
            f"FP: {brake_fp} | "
            f"FN: {brake_fn}"
        )

        print("=" * 80)

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                CHECKPOINT
            )

            print(
                f">>> Validation loss decreased! "
                f"Saved {CHECKPOINT} <<<"
            )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 80)

    print(
        "TRAINING COMPLETE"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Checkpoint: {CHECKPOINT}"
    )

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # DATA PATHS
    # ========================================================

    BASE_PATHS = [
        "./PDM_Lite_Carla_LB2_Data/Town01/data/*",
        "./PDM_Lite_Carla_LB2_Data/Town02/data/*",
        "./PDM_Lite_Carla_LB2_Data/Town03/data/*/*",
    ]

    # ========================================================
    # RANDOM SEEDS
    # ========================================================

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 80)

    print(
        "PDM-LITE NEURAL DRIVER TRAINING V8"
    )

    print("=" * 80)

    print(
        f"Device: {device}"
    )

    print(
        f"Telemetry: {NUM_TELEMETRY}"
    )

    print(
        f"Checkpoint: {CHECKPOINT}"
    )

    print(
        "Steering strategy:"
    )

    print(
        "  sqrt inverse-frequency sampling"
    )

    print(
        "  Huber steering loss"
    )

    print(
        "Brake input:"
    )

    print(
        "  REMOVED"
    )

    print(
        "Brake:"
    )

    print(
        "  target only"
    )

    print(
        "Lateral distance:"
    )

    print(
        "  route-based"
    )

    print(
        "  normalized to [-1, 1]"
    )

    print("=" * 80)

    # ========================================================
    # BUILD INDEX
    # ========================================================

    master_list = build_path_index(
        BASE_PATHS
    )

    print(
        "Total indexed samples:",
        len(master_list)
    )

    if len(master_list) == 0:

        raise RuntimeError(
            "No training samples found."
        )

    # ========================================================
    # DATASET DISTRIBUTION
    # ========================================================

    summarize_steering_distribution(
        master_list
    )

    # ========================================================
    # SHUFFLE
    # ========================================================

    random.shuffle(
        master_list
    )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    val_size = int(
        len(master_list)
        *
        VAL_RATIO
    )

    val_list = master_list[
        :val_size
    ]

    train_list = master_list[
        val_size:
    ]

    print()
    print(
        "Training samples:",
        len(train_list)
    )

    print(
        "Validation samples:",
        len(val_list)
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = CarlaDataset(
        train_list
    )

    val_dataset = CarlaDataset(
        val_list
    )

    # ========================================================
    # DATASET CHECK
    # ========================================================

    print()
    print(
        "Checking dataset interface..."
    )

    sample_image, sample_telemetry, sample_target = (
        train_dataset[0]
    )

    print(
        "Image shape:",
        tuple(sample_image.shape)
    )

    print(
        "Telemetry shape:",
        tuple(sample_telemetry.shape)
    )

    print(
        "Target shape:",
        tuple(sample_target.shape)
    )

    # --------------------------------------------------------
    # IMAGE CHECK
    # --------------------------------------------------------

    if tuple(sample_image.shape) != (
        3,
        200,
        400
    ):

        raise RuntimeError(
            "Unexpected image shape: "
            f"{sample_image.shape}"
        )

    # --------------------------------------------------------
    # TELEMETRY CHECK
    # --------------------------------------------------------

    if sample_telemetry.numel() != NUM_TELEMETRY:

        raise RuntimeError(
            "Dataset must return exactly "
            f"{NUM_TELEMETRY} telemetry values."
        )

    # --------------------------------------------------------
    # TARGET CHECK
    # --------------------------------------------------------

    if sample_target.numel() != 3:

        raise RuntimeError(
            "Dataset must return exactly "
            "3 targets."
        )

    print(
        "Dataset interface: OK"
    )

    # ========================================================
    # STEERING SAMPLING
    # ========================================================

    sample_weights, train_counts = (
        get_steering_sampling_weights(
            train_list
        )
    )

    print()
    print(
        "Training steering buckets:"
    )

    for name in (
        "strong_left",
        "left",
        "straight",
        "right",
        "strong_right"
    ):

        print(
            f"{name:>13}: "
            f"{train_counts[name]:6d}"
        )

    sample_weights = torch.tensor(
        sample_weights,
        dtype=torch.double
    )

    # ========================================================
    # WEIGHTED SAMPLER
    # ========================================================

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_list),
        replacement=True
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = AutonomousDriver().to(
        device
    )

    # ========================================================
    # PARAMETER COUNT
    # ========================================================

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        "Model parameters:",
        f"{parameter_count:,}"
    )

    print(
        "Fusion input features:",
        model.fusion_branch[
            0
        ].in_features
    )

    print(
        "Outputs:"
    )

    print(
        "  0 = Steering"
    )

    print(
        "  1 = Throttle"
    )

    print(
        "  2 = Brake logit"
    )

    # ========================================================
    # TRAIN
    # ========================================================

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )