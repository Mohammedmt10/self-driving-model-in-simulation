import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import (CarlaDataset, build_route_split,
                     summarize_steering_distribution,
                     get_steering_sampling_weights,
                     compute_loss_weights)


BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 3e-4
SEED = 42

CHECKPOINT = "model_v14.pth"


class AutonomousDriver(nn.Module):

    def __init__(self):
        super().__init__()
        self.vision_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveMaxPool2d((5, 5)),
            nn.Flatten(),          # 256 * 5 * 5 = 6400
            nn.Dropout(0.3),
            nn.Linear(6400, 512),
            nn.ReLU(),
        )
        self.vision_out_dim = 512

        self.vehicle_state_branch = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.RMSNorm(128),
            nn.Dropout(0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.RMSNorm(128),
            nn.Dropout(0.4),
        )

        self.junction_state_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.RMSNorm(16),
            nn.Dropout(0.4),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.RMSNorm(16),
            nn.Dropout(0.4),
        )

        self.command_embedding = nn.Embedding(7, 16)
        self.next_command_embedding = nn.Embedding(7, 16)

        # Fusion: vision(512) + vehicle(128) + junction(16) + command(16)
        #         + next_command(16) = 688
        self.fusion_branch = nn.Sequential(
            nn.Linear(512 + 128 + 16 + 16 + 16, 128),
            nn.ReLU(),
            nn.RMSNorm(128),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.RMSNorm(128),
            nn.Dropout(0.5),
        )

        self.steering_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16),
            nn.Linear(16,1),
            nn.Tanh(),
        )

        self.throttle_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16),
            nn.Linear(16,1),
            nn.Sigmoid(),
        )

        self.brake_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 2),

        )

    def forward(self, image_input, telemetry_input):

        speed = telemetry_input[:, 0:1]
        command = telemetry_input[:, 1].long().clamp(1, 6)
        junction = telemetry_input[:, 2:3]
        speed_limit = telemetry_input[:, 3:4]
        angle = telemetry_input[:, 4:5]
        next_command = telemetry_input[:, 5].long().clamp(1, 6)
        theta = telemetry_input[:, 6:7]
        lateral_distance = telemetry_input[:, 7:8]

        vehicle_state = torch.cat(
            (speed, speed_limit, angle, theta, lateral_distance), dim=1
        )
        vehicle_features = self.vehicle_state_branch(vehicle_state)

        junction_features = self.junction_state_branch(junction)

        command_features = self.command_embedding(command)
        next_command_features = self.next_command_embedding(next_command)

        vision_features = self.vision_branch(image_input)

        combined_features = torch.cat(
            (
                vision_features,
                vehicle_features,
                junction_features,
                command_features,
                next_command_features,
            ),
            dim=1,
        )

        fused_features = self.fusion_branch(combined_features)

        steering = self.steering_head(fused_features)
        throttle = self.throttle_head(fused_features)
        brake_logits = self.brake_head(fused_features)

        return steering, throttle, brake_logits


if __name__ == "__main__":

    # Route-based split across ALL towns: for each (town, scenario)
    # the routes are sorted and split ~80/20 (first 80% train, last
    # 20% validation), so validation holds out entire held-out routes
    # rather than whole towns.
    DATA_DIR = "./PDM_Lite_Carla_LB2_Data"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_list, val_list = build_route_split(DATA_DIR, val_fraction=0.2)

    print(f"Training samples: {len(train_list)}")
    print(f"Validation samples: {len(val_list)}")

    summarize_steering_distribution(train_list)
    print("--- Validation ---")
    summarize_steering_distribution(val_list)

    # --------------------------------------------------------
    # Per-output loss weights, computed directly from the training
    # data (no assumed scales).
    #
    # `compute_loss_weights` returns the baseline loss a trivial
    # constant predictor achieves on each task:
    #   steer    -> MSE of predicting the mean steer  (= variance)
    #   throttle -> MSE of predicting the mean throttle (= variance)
    #   brake    -> cross-entropy of the marginal class dist
    #
    # We weight each task by the INVERSE of its baseline, so every
    # term starts contributing ~1.0 to the weighted total.
    # --------------------------------------------------------
    loss_weights = compute_loss_weights(train_list)
    w1 = 1.0 / loss_weights["steer"]
    w2 = 1.0 / loss_weights["throttle"]
    w3 = 1.0 / loss_weights["brake"]
    print("\nOutput weights (inverse of data-derived baseline loss):")
    print(f"  w1 (steer)    = 1/{loss_weights['steer']:.6f} = {w1:.3f}")
    print(f"  w2 (throttle) = 1/{loss_weights['throttle']:.6f} = {w2:.3f}")
    print(f"  w3 (brake)    = 1/{loss_weights['brake']:.6f} = {w3:.3f} "
          f"(brake rate {loss_weights['counts']['p_brake']:.3f}, "
          f"n={loss_weights['counts']['n']})")

    train_dataset = CarlaDataset(train_list)
    val_dataset = CarlaDataset(val_list)

    # WeightedRandomSampler for balanced steering distribution
    sample_weights, train_counts = get_steering_sampling_weights(train_list)
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_list),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=8,
        sampler=sampler,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

    model = AutonomousDriver().to(device)
    print(f"Training device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Vision output features: {model.vision_out_dim}")

    # Single optimizer over all parameters. Heavier weight decay on
    # the shared backbone to regularize the memorization that made
    # the brake head overfit; lighter on the task heads.
    shared_params = (
        list(model.vision_branch.parameters())
        + list(model.vehicle_state_branch.parameters())
        + list(model.junction_state_branch.parameters())
        + list(model.command_embedding.parameters())
        + list(model.next_command_embedding.parameters())
        + list(model.fusion_branch.parameters())
    )
    head_params = (
        list(model.steering_head.parameters())
        + list(model.throttle_head.parameters())
        + list(model.brake_head.parameters())
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": shared_params, "weight_decay": 1e-3},
            {"params": head_params, "weight_decay": 1e-4},
        ],
        lr=LEARNING_RATE,
    )

    # One ReduceLROnPlateau over the single weighted validation loss.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    mse_criterion = nn.MSELoss()

    ce_criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Loss normalization via fixed output weights.
    #
    # Each task's raw loss sits on a different scale. Instead of
    # assuming those scales (or tracking an EMA), we divide each
    # raw loss by the baseline loss a trivial constant predictor
    # achieves on the training data (computed above in
    # `loss_weights`), so every task contributes ~1.0 to the
    # total at the start of training.
    # --------------------------------------------------------

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0.0
        running_steer_loss = 0.0
        running_throttle_loss = 0.0
        running_brake_loss = 0.0
        # Raw (un-normalized) losses for progress tracking.
        running_raw_steer = 0.0
        running_raw_throttle = 0.0
        running_raw_brake = 0.0
        train_samples = 0

        for batch_idx, (images, telemetry, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            telemetry = telemetry.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            steering, throttle, brake_logits = model(images, telemetry)

            # Per-output losses, computed separately.
            l_steer = mse_criterion(steering.squeeze(-1), targets[:, 0])
            l_throttle = mse_criterion(throttle.squeeze(-1), targets[:, 1])
            l_brake = ce_criterion(brake_logits, targets[:, 2].long())

            # Weighted scalar total: total = w1*l_steer + w2*l_throttle + w3*l_brake.
            steer_term = w1 * l_steer
            throttle_term = w2 * l_throttle
            brake_term = w3 * l_brake
            total = steer_term + throttle_term + brake_term

            # Single backward pass on the weighted total.
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            batch_size = images.size(0)
            running_loss += total.item() * batch_size
            running_steer_loss += steer_term.item() * batch_size
            running_throttle_loss += throttle_term.item() * batch_size
            running_brake_loss += brake_term.item() * batch_size
            running_raw_steer += l_steer.item() * batch_size
            running_raw_throttle += l_throttle.item() * batch_size
            running_raw_brake += l_brake.item() * batch_size
            train_samples += batch_size

            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / max(train_samples, 1)
                avg_steer = running_steer_loss / max(train_samples, 1)
                avg_throttle = running_throttle_loss / max(train_samples, 1)
                avg_brake = running_brake_loss / max(train_samples, 1)
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Avg Train Loss: {avg_loss:.4f} (Steer: {avg_steer:.4f}, Throttle: {avg_throttle:.4f}, Brake: {avg_brake:.4f})")

        train_loss = running_loss / max(train_samples, 1)
        train_steer_loss = running_steer_loss / max(train_samples, 1)
        train_throttle_loss = running_throttle_loss / max(train_samples, 1)
        train_brake_loss = running_brake_loss / max(train_samples, 1)
        train_raw_steer = running_raw_steer / max(train_samples, 1)
        train_raw_throttle = running_raw_throttle / max(train_samples, 1)
        train_raw_brake = running_raw_brake / max(train_samples, 1)

        model.eval()
        val_loss_total = 0.0
        val_steer_loss = 0.0
        val_throttle_loss = 0.0
        val_brake_loss = 0.0
        # Raw (un-normalized) val losses.
        val_raw_steer = 0.0
        val_raw_throttle = 0.0
        val_raw_brake = 0.0
        val_samples = 0
        # Brake prediction accuracy counters (whole batch at a time).
        val_brake_correct = 0
        val_brake_total = 0

        with torch.no_grad():
            for images, telemetry, targets in val_loader:
                images = images.to(device, non_blocking=True)
                telemetry = telemetry.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                steering, throttle, brake_logits = model(images, telemetry)

                l_steer = mse_criterion(steering.squeeze(-1), targets[:, 0])
                l_throttle = mse_criterion(throttle.squeeze(-1), targets[:, 1])
                l_brake = ce_criterion(brake_logits, targets[:, 2].long())

                # Weighted scalar total: w1*l_steer + w2*l_throttle + w3*l_brake.
                steer_term = w1 * l_steer
                throttle_term = w2 * l_throttle
                brake_term = w3 * l_brake
                total = steer_term + throttle_term + brake_term

                # Count correct brake-class predictions.
                brake_pred = brake_logits.argmax(dim=1)
                val_brake_correct += (
                    brake_pred == targets[:, 2].long()
                ).sum().item()
                val_brake_total += images.size(0)

                batch_size = images.size(0)
                val_loss_total += total.item() * batch_size
                val_steer_loss += steer_term.item() * batch_size
                val_throttle_loss += throttle_term.item() * batch_size
                val_brake_loss += brake_term.item() * batch_size
                val_raw_steer += l_steer.item() * batch_size
                val_raw_throttle += l_throttle.item() * batch_size
                val_raw_brake += l_brake.item() * batch_size
                val_samples += batch_size

        val_loss = val_loss_total / max(val_samples, 1)
        val_steer = val_steer_loss / max(val_samples, 1)
        val_throttle = val_throttle_loss / max(val_samples, 1)
        val_brake = val_brake_loss / max(val_samples, 1)
        val_brake_acc = val_brake_correct / max(val_brake_total, 1)
        val_raw_steer_avg = val_raw_steer / max(val_samples, 1)
        val_raw_throttle_avg = val_raw_throttle / max(val_samples, 1)
        val_raw_brake_avg = val_raw_brake / max(val_samples, 1)

        # Raw losses show actual learning progress (the weighted total
        # reads ~3.0 per sample by design, ~1.0 per task at start).
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} "
              f"(Steer: {train_steer_loss:.4f}, Throttle: {train_throttle_loss:.4f}, Brake: {train_brake_loss:.4f}) | "
              f"Val Loss: {val_loss:.4f} "
              f"(Steer: {val_steer:.4f}, Throttle: {val_throttle:.4f}, Brake: {val_brake:.4f})")
        print(f"  Raw losses — "
              f"Train: steer={train_raw_steer:.6f} throttle={train_raw_throttle:.6f} brake={train_raw_brake:.4f} | "
              f"Val: steer={val_raw_steer_avg:.6f} throttle={val_raw_throttle_avg:.6f} brake={val_raw_brake_avg:.4f}")
        print(f"  Val brake accuracy: {val_brake_correct}/{val_brake_total} = {val_brake_acc:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  Saved {CHECKPOINT}")

    print(f"\nBest Val Loss: {best_val_loss:.4f}")
    print(f"Vision output features: {model.vision_out_dim}")
