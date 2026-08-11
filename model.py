import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CarlaDataset, build_path_index


class AutonomousDriver(nn.Module):
    def __init__(self):
        super().__init__()

        self.vision_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False, padding_mode='zeros'),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=False, padding_mode='zeros'),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False, padding_mode='zeros'),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='zeros'),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='zeros'),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((5, 5)),
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 256),
            nn.SiLU(),
            nn.Linear(256, 200),
            nn.SiLU(),
            nn.Dropout(0.2)
        )

        self.telemetry_branch = nn.Sequential(
            nn.Linear(8, 32),
            nn.SiLU(),
            nn.Linear(32, 64),
            nn.SiLU(),
        )

        self.fusion_branch = nn.Sequential(
            nn.Linear(200 + 64, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 3)
        )

    def forward(self, image_input, telemetry_input):

        vision_features = self.vision_branch(image_input)
        telemetry_features = self.telemetry_branch(telemetry_input)

        combined_features = torch.cat(
            (vision_features, telemetry_features),
            dim=1
        )

        output = self.fusion_branch(combined_features)

        return output


def train_model(master_list):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    model = AutonomousDriver().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=(1e-4) / 2
    )

    # ============================================================
    # LOSS
    # ============================================================

    criterion = nn.HuberLoss(
        reduction="none"
    )

    # Steering, throttle, brake
    output_weights = torch.tensor(
        [1.0, 1.0, 2.0],
        dtype=torch.float32,
        device=device
    )

    # ============================================================
    # TOWN-BASED SPLIT
    #
    # TRAIN:
    #   Town01
    #   Town03
    #
    # VALIDATION:
    #   Town02
    # ============================================================

    train_master_list = build_path_index([
        "./PDM_Lite_Carla_LB2_Data/Town01/data/*",
        "./PDM_Lite_Carla_LB2_Data/Town03/data/*/*",
    ])

    val_master_list = build_path_index([
        "./PDM_Lite_Carla_LB2_Data/Town02/data/*",
    ])

    print(
        f"Training samples: {len(train_master_list)}"
    )

    print(
        f"Validation samples: {len(val_master_list)}"
    )

    if len(train_master_list) == 0:
        raise ValueError(
            "Training dataset is empty."
        )

    if len(val_master_list) == 0:
        raise ValueError(
            "Validation dataset is empty."
        )

    train_dataset = CarlaDataset(
        train_master_list
    )

    val_dataset = CarlaDataset(
        val_master_list
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    epochs = 10

    best_val_loss = float("inf")

    print("Starting training...")

    for epoch in range(epochs):

        # ========================================================
        # TRAINING
        # ========================================================

        model.train()

        window_loss = 0.0
        epoch_loss = 0.0

        total_train_samples = 0

        train_output_error = torch.zeros(
            3,
            device=device
        )

        for i, (
            images,
            telemetry,
            targets
        ) in enumerate(train_dataloader):

            images = images.to(device)
            telemetry = telemetry.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(
                images,
                telemetry
            )

            # ----------------------------------------------------
            # Individual output losses
            # ----------------------------------------------------

            individual_loss = criterion(
                outputs,
                targets
            )

            # ----------------------------------------------------
            # Weighted loss
            # ----------------------------------------------------

            weighted_loss = (
                individual_loss
                * output_weights
            )

            loss = weighted_loss.mean()

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            # ----------------------------------------------------
            # MAE for each output
            # ----------------------------------------------------

            absolute_error = torch.abs(
                outputs - targets
            )

            train_output_error += (
                absolute_error.sum(
                    dim=0
                )
            )

            # ----------------------------------------------------
            # Loss tracking
            # ----------------------------------------------------

            batch_size = images.size(0)

            window_loss += (
                loss.item()
                * batch_size
            )

            epoch_loss += (
                loss.item()
                * batch_size
            )

            total_train_samples += batch_size

            if (i + 1) % 100 == 0:

                print(
                    f"[Epoch {epoch + 1}, "
                    f"Batch {i + 1}] "
                    f"train_loss: "
                    f"{window_loss / (100 * 32):.3f}"
                )

                window_loss = 0.0

        avg_train_loss = (
            epoch_loss
            / total_train_samples
        )

        train_mae = (
            train_output_error
            / total_train_samples
        )

        # ========================================================
        # VALIDATION
        # ========================================================

        model.eval()

        val_loss = 0.0

        total_val_samples = 0

        val_output_error = torch.zeros(
            3,
            device=device
        )

        val_output_loss = torch.zeros(
            3,
            device=device
        )

        with torch.no_grad():

            for (
                images,
                telemetry,
                targets
            ) in val_dataloader:

                images = images.to(device)
                telemetry = telemetry.to(device)
                targets = targets.to(device)

                outputs = model(
                    images,
                    telemetry
                )

                # ------------------------------------------------
                # Individual Huber losses
                # ------------------------------------------------

                individual_loss = criterion(
                    outputs,
                    targets
                )

                # ------------------------------------------------
                # Weighted total loss
                # ------------------------------------------------

                weighted_loss = (
                    individual_loss
                    * output_weights
                )

                loss = weighted_loss.mean()

                # ------------------------------------------------
                # Absolute error
                # ------------------------------------------------

                absolute_error = torch.abs(
                    outputs - targets
                )

                batch_size = images.size(0)

                val_output_error += (
                    absolute_error.sum(
                        dim=0
                    )
                )

                val_output_loss += (
                    individual_loss.sum(
                        dim=0
                    )
                )

                val_loss += (
                    loss.item()
                    * batch_size
                )

                total_val_samples += batch_size

        # ========================================================
        # FINAL VALIDATION METRICS
        # ========================================================

        avg_val_loss = (
            val_loss
            / total_val_samples
        )

        val_mae = (
            val_output_error
            / total_val_samples
        )

        val_individual_loss = (
            val_output_loss
            / total_val_samples
        )

        # ========================================================
        # PRINT RESULTS
        # ========================================================

        print(
            f"--- End of Epoch {epoch + 1} "
            f"| Train Loss: {avg_train_loss:.4f} "
            f"| Val Loss: {avg_val_loss:.4f} ---"
        )

        print(
            f"    Train MAE | "
            f"Steer: {train_mae[0].item():.4f} | "
            f"Throttle: {train_mae[1].item():.4f} | "
            f"Brake: {train_mae[2].item():.4f}"
        )

        print(
            f"    Val MAE   | "
            f"Steer: {val_mae[0].item():.4f} | "
            f"Throttle: {val_mae[1].item():.4f} | "
            f"Brake: {val_mae[2].item():.4f}"
        )

        print(
            f"    Val Huber | "
            f"Steer: {val_individual_loss[0].item():.4f} | "
            f"Throttle: {val_individual_loss[1].item():.4f} | "
            f"Brake: {val_individual_loss[2].item():.4f}"
        )

        # ========================================================
        # BEST MODEL
        # ========================================================

        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss

            torch.save(
                model.state_dict(),
                "carla_driver_pytorch_v1_best.pth"
            )

            print(
                ">>> Validation loss decreased! "
                "New best model saved. <<<\n"
            )

        else:

            print("")

    # ============================================================
    # FINAL MODEL
    # ============================================================

    torch.save(
        model.state_dict(),
        "carla_driver_pytorch_v1_final.pth"
    )

    print("Training complete!")


if __name__ == "__main__":

    master_list = build_path_index([
        "./PDM_Lite_Carla_LB2_Data/Town01/data/*",
        "./PDM_Lite_Carla_LB2_Data/Town02/data/*",
        "./PDM_Lite_Carla_LB2_Data/Town03/data/*/*",
    ])

    train_model(master_list)