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
# MODEL
# ============================================================

class AutonomousDriver(nn.Module):

    def __init__(self):

        super().__init__()

        # ====================================================
        # VISION
        # ====================================================

        self.vision_branch = nn.Sequential(
            nn.Conv2d(3,32, kernel_size=5, stride=2, padding=2, bias=True, padding_mode="zeros"),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=True, padding_mode="zeros"),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=True, padding_mode="zeros"),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="zeros"),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="zeros"),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="zeros"),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((10, 10)),
            nn.Flatten(),
            nn.Linear(128 * 10 * 10, 256),
            nn.ReLU(),
            nn.Linear(256, 500),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # ====================================================
        # VEHICLE STATE
        #
        # speed
        # speed_limit
        # angle
        # theta
        # ====================================================

        self.vehicle_state_branch = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True)
        )

        # ====================================================
        # JUNCTION
        # ====================================================

        self.junction_state_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.ReLU(inplace=True)
        )

        # ====================================================
        # COMMAND
        # ====================================================

        self.command_embedding = nn.Embedding(7, 32)

        # ====================================================
        # NEXT COMMAND
        # ====================================================

        self.next_command_embedding = nn.Embedding(7, 32)

        # ====================================================
        # OBJECT TYPE
        # ====================================================

        self.object_type_embedding = nn.Embedding(5, 16)

        # ====================================================
        # OBJECT STATE
        #
        # 16 object embedding
        # + 1 object id
        # + 1 object distance
        # = 18
        # ====================================================

        self.object_state_branch = nn.Sequential(
            nn.Linear(18, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True)
        )

        # ====================================================
        # FUSION
        #
        # 64 vehicle
        # 32 junction
        # 32 command
        # 32 next command
        # 64 object
        # 500 vision
        # ----------------
        # = 724
        # ====================================================

        self.fusion_branch = nn.Sequential(
            nn.Linear(724, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # ====================================================
        # STEERING
        # ====================================================

        self.steering_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Tanh()
        )

        # ====================================================
        # THROTTLE
        # ====================================================

        self.throttle_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # ====================================================
        # BRAKE
        #
        # Brake is still a TARGET.
        # It is NOT an input anymore.
        # ====================================================

        self.brake_head = nn.Linear(128, 1)

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, image_input, telemetry_input):
        if telemetry_input.ndim != 2:
            raise RuntimeError(
                "Telemetry must be [batch, 10]. "
                f"Got {telemetry_input.shape}"
            )

        if telemetry_input.shape[1] != 10:
            raise RuntimeError(
                "Model expects exactly 10 telemetry values. "
                f"Got {telemetry_input.shape[1]}"
            )

        # ====================================================
        # TELEMETRY
        #
        # 0 speed
        # 1 command
        # 2 junction
        # 3 object_type
        # 4 object_id
        # 5 object_distance
        # 6 speed_limit
        # 7 angle
        # 8 next_command
        # 9 theta
        # ====================================================

        speed = (telemetry_input[:, 0:1])
        command = (telemetry_input[:, 1].long())
        junction = (telemetry_input[:, 2:3])
        object_type = (telemetry_input[:, 3].long())
        object_id = (telemetry_input[:, 4:5])
        object_distance = (telemetry_input[:, 5:6])
        speed_limit = (telemetry_input[:, 6:7])
        angle = (telemetry_input[:, 7:8])
        next_command = (telemetry_input[:, 8].long())
        theta = (telemetry_input[:, 9:10])

        # ====================================================
        # VEHICLE
        # ====================================================

        vehicle_state = torch.cat(
            (
                speed,
                speed_limit,
                angle,
                theta
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
        # OBJECT
        # ====================================================

        object_type_features = (
            self.object_type_embedding(
                object_type
            )
        )

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
        # VISION
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
                "Fusion size mismatch. "
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

        steering_output = (
            self.steering_head(
                fused_features
            )
        )

        throttle_output = (
            self.throttle_head(
                fused_features
            )
        )

        brake_output = (
            self.brake_head(
                fused_features
            )
        )

        return torch.cat(
            (
                steering_output,
                throttle_output,
                brake_output
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    regression_criterion = nn.HuberLoss(
        reduction="none"
    )

    brake_criterion = nn.BCEWithLogitsLoss(
        reduction="none"
    )

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
    # EPOCHS
    # ========================================================

    for epoch in range(epochs):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        running_loss = 0.0

        train_absolute_error = torch.zeros(
            3,
            device=device
        )

        train_samples = 0

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

            if telemetry.shape[1] != 10:

                raise RuntimeError(
                    "Training dataset returned "
                    f"{telemetry.shape[1]} telemetry values. "
                    "Expected 10."
                )

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

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(
                images,
                telemetry
            )

            # ------------------------------------------------
            # STEERING
            # ------------------------------------------------

            steer_loss = (
                regression_criterion(
                    outputs[:, 0],
                    targets[:, 0]
                )
            )

            # ------------------------------------------------
            # THROTTLE
            # ------------------------------------------------

            throttle_loss = (
                regression_criterion(
                    outputs[:, 1],
                    targets[:, 1]
                )
            )

            # ------------------------------------------------
            # BRAKE
            #
            # Target only.
            # No brake input.
            # ------------------------------------------------

            brake_loss = (
                brake_criterion(
                    outputs[:, 2],
                    targets[:, 2]
                )
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

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            batch_size = images.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            with torch.no_grad():

                brake_probability = (
                    torch.sigmoid(
                        outputs[:, 2]
                    )
                )

                brake_prediction = (
                    brake_probability >= 0.5
                ).float()

                batch_error = torch.stack(
                    (
                        torch.abs(
                            outputs[:, 0]
                            - targets[:, 0]
                        ),

                        torch.abs(
                            outputs[:, 1]
                            - targets[:, 1]
                        ),

                        torch.abs(
                            brake_prediction
                            - targets[:, 2]
                        )
                    ),
                    dim=1
                )

                train_absolute_error += (
                    batch_error.sum(dim=0)
                )

            train_samples += batch_size

            if batch_idx % 100 == 0:

                print(
                    f"[Epoch {epoch + 1}, "
                    f"Batch {batch_idx}] "
                    f"train_loss: "
                    f"{running_loss / train_samples:.4f}"
                )

        train_loss = (
            running_loss / train_samples
        )

        train_mae = (
            train_absolute_error
            / train_samples
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

        brake_tp = 0
        brake_tn = 0
        brake_fp = 0
        brake_fn = 0

        steer_sign_correct = 0
        steer_sign_total = 0

        strong_steer_error = 0.0
        strong_steer_samples = 0

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

                if telemetry.shape[1] != 10:

                    raise RuntimeError(
                        "Validation dataset returned "
                        f"{telemetry.shape[1]} values. "
                        "Expected 10."
                    )

                outputs = model(
                    images,
                    telemetry
                )

                steer_loss = (
                    regression_criterion(
                        outputs[:, 0],
                        targets[:, 0]
                    )
                )

                throttle_loss = (
                    regression_criterion(
                        outputs[:, 1],
                        targets[:, 1]
                    )
                )

                brake_loss = (
                    brake_criterion(
                        outputs[:, 2],
                        targets[:, 2]
                    )
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
                    loss.item() * batch_size
                )

                val_output_loss += (
                    individual_loss.sum(dim=0)
                )

                # ------------------------------------------------
                # Predictions
                # ------------------------------------------------

                brake_probability = (
                    torch.sigmoid(
                        outputs[:, 2]
                    )
                )

                brake_prediction = (
                    brake_probability >= 0.5
                ).float()

                batch_error = torch.stack(
                    (
                        torch.abs(
                            outputs[:, 0]
                            - targets[:, 0]
                        ),

                        torch.abs(
                            outputs[:, 1]
                            - targets[:, 1]
                        ),

                        torch.abs(
                            brake_prediction
                            - targets[:, 2]
                        )
                    ),
                    dim=1
                )

                val_absolute_error += (
                    batch_error.sum(dim=0)
                )

                # ------------------------------------------------
                # Brake confusion
                # ------------------------------------------------

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

                # ------------------------------------------------
                # Steering sign
                # ------------------------------------------------

                meaningful = (
                    torch.abs(
                        targets[:, 0]
                    ) >= 0.05
                )

                if meaningful.any():

                    true_sign = torch.sign(
                        targets[:, 0][meaningful]
                    )

                    pred_sign = torch.sign(
                        outputs[:, 0][meaningful]
                    )

                    steer_sign_correct += (
                        (true_sign == pred_sign)
                        .sum()
                        .item()
                    )

                    steer_sign_total += (
                        meaningful.sum().item()
                    )

                # ------------------------------------------------
                # Strong turns
                # ------------------------------------------------

                strong = (
                    torch.abs(
                        targets[:, 0]
                    ) >= 0.25
                )

                if strong.any():

                    error = torch.abs(
                        outputs[:, 0][strong]
                        -
                        targets[:, 0][strong]
                    )

                    strong_steer_error += (
                        error.sum().item()
                    )

                    strong_steer_samples += (
                        strong.sum().item()
                    )

                val_samples += batch_size

        # ====================================================
        # METRICS
        # ====================================================

        val_loss = (
            val_loss_total / val_samples
        )

        val_mae = (
            val_absolute_error
            / val_samples
        )

        val_output_loss_avg = (
            val_output_loss
            / val_samples
        )

        brake_total = (
            brake_tp
            + brake_tn
            + brake_fp
            + brake_fn
        )

        brake_accuracy = (
            (brake_tp + brake_tn)
            / brake_total
            if brake_total > 0
            else 0.0
        )

        brake_precision = (
            brake_tp
            / (brake_tp + brake_fp)
            if (brake_tp + brake_fp) > 0
            else 0.0
        )

        brake_recall = (
            brake_tp
            / (brake_tp + brake_fn)
            if (brake_tp + brake_fn) > 0
            else 0.0
        )

        steer_sign_accuracy = (
            steer_sign_correct
            / steer_sign_total
            if steer_sign_total > 0
            else 0.0
        )

        strong_turn_mae = (
            strong_steer_error
            / strong_steer_samples
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
            f"Sign Accuracy: {steer_sign_accuracy:.4f} | "
            f"Strong-turn MAE: {strong_turn_mae:.4f}"
        )

        print(
            f"Brake Metrics | "
            f"Accuracy: {brake_accuracy:.4f} | "
            f"Precision: {brake_precision:.4f} | "
            f"Recall: {brake_recall:.4f}"
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
        # SAVE BEST
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                "model_v5.pth"
            )

            print(
                ">>> Validation loss decreased! "
                "Saved model_v5.pth <<<"
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
        "Checkpoint: model_v5.pth"
    )

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    BASE_PATHS = [
        "./PDM_Lite_Carla_LB2_Data/Town01/data/*",
        "./PDM_Lite_Carla_LB2_Data/Town02/data/*",
        "./PDM_Lite_Carla_LB2_Data/Town03/data/*/*",
    ]

    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 2e-5
    VAL_RATIO = 0.15
    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 80)

    print(
        "PDM-LITE NEURAL DRIVER TRAINING V5"
    )

    print("=" * 80)

    print(
        "Device:",
        device
    )

    print(
        "Telemetry:",
        10
    )

    print(
        "Checkpoint:",
        "model_v5.pth"
    )

    print(
        "Steering strategy:"
    )

    print(
        "  sqrt inverse-frequency sampling"
    )

    print(
        "  normal Huber steering loss"
    )

    print(
        "Brake input:"
    )

    print(
        "  REMOVED"
    )

    print("=" * 80)

    # ========================================================
    # INDEX
    # ========================================================

    master_list = build_path_index(
        BASE_PATHS
    )

    print(
        "Total indexed samples:",
        len(master_list)
    )

    if not master_list:

        raise RuntimeError(
            "No training samples found."
        )

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    summarize_steering_distribution(
        master_list
    )

    # ========================================================
    # SPLIT
    # ========================================================

    random.shuffle(
        master_list
    )

    val_size = int(
        len(master_list)
        * VAL_RATIO
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

    if sample_telemetry.numel() != 10:

        raise RuntimeError(
            "Dataset must return exactly "
            "10 telemetry values."
        )

    if sample_image.shape != (
        3,
        200,
        400
    ):

        raise RuntimeError(
            "Unexpected image shape."
        )

    if sample_target.numel() != 3:

        raise RuntimeError(
            "Expected target shape [3]."
        )

    print(
        "Dataset interface: OK"
    )

    # ========================================================
    # SAMPLER
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
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = AutonomousDriver().to(device)

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
        model.fusion_branch[0].in_features
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