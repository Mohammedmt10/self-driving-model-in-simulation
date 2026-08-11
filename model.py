import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from dataset import CarlaDataset, build_path_index

class AutonomousDriver(nn.Module):
    def __init__(self):
        super().__init__()

        self.vision_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((20, 20)),
            nn.Flatten(),
            nn.Linear(128 * 20 * 20, 200),
            nn.SiLU(),
            nn.Dropout(0.2)
        )

        self.telemetry_branch = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, 64),
            nn.SiLU(),
        )

        self.fusion_branch = nn.Sequential(
            nn.Linear(200 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Output: steering angle, throttle and brake
        )

    def forward(self, image_input, telemetry_input):

        vision_features = self.vision_branch(image_input)
        telemetry_features = self.telemetry_branch(telemetry_input)

        combined_features = torch.cat((vision_features, telemetry_features), dim=1)

        output = self.fusion_branch(combined_features)
        return output


def train_model(master_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AutonomousDriver().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.HuberLoss()

    dataset = CarlaDataset(master_list)

    # --- ADDED: Split dataset into 80% train and 20% validation ---
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Safety check in case the dataset is too small
    if val_size == 0:
        raise ValueError("Dataset is too small to split into training and validation sets.")

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # --- ADDED: Separate dataloaders ---
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    epochs = 10

    # FIX 3: Track the best validation loss
    best_val_loss = float('inf')

    print("Starting training...")

    for epoch in range(epochs):
        # Training Phase
        model.train()
        window_loss = 0.0
        epoch_loss = 0.0
        total_train_samples = 0

        for i, (images, telemetry, targets) in enumerate(train_dataloader):
            images = images.to(device)
            telemetry = telemetry.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(images, telemetry)
            loss = criterion(outputs, targets)

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )
            optimizer.step()

            # FIX 1: Multiply loss.item() (mean of batch) by the actual batch size
            batch_size = images.size(0)
            window_loss += loss.item() * batch_size
            epoch_loss += loss.item() * batch_size
            total_train_samples += batch_size

            if (i + 1) % 100 == 0:
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] train_loss: {window_loss / (100 * 32):.3f}")
                window_loss = 0.0

        avg_train_loss = epoch_loss / total_train_samples

        # --- MODIFIED: Validation Phase ---
        model.eval()
        val_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for images, telemetry, targets in val_dataloader:
                images = images.to(device)
                telemetry = telemetry.to(device)
                targets = targets.to(device)

                outputs = model(images, telemetry)
                loss = criterion(outputs, targets)

                # FIX 1: Exact loss tracking
                batch_size = images.size(0)
                val_loss += loss.item() * batch_size
                total_val_samples += batch_size

        # FIX 2: Guard against zero-division, divide by exact sample count
        avg_val_loss = val_loss / total_val_samples if total_val_samples > 0 else 0.0

        print(f"--- End of Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ---")

        # FIX 3: Save the model ONLY if validation loss improved (Model Checkpointing)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "carla_driver_pytorch_v4_best.pth")
            print(">>> Validation loss decreased! New best model saved. <<<\n")
        else:
            print("")

    # Optional: Save the final epoch regardless (so you have both best and final states)
    torch.save(model.state_dict(), "carla_driver_pytorch_v4_final.pth")
    print("Training complete!")


if __name__ == "__main__":
    master_list = build_path_index([
        "./PDM_Lite_Carla_LB2_Data/Town01/data/*",
        "./PDM_Lite_Carla_LB2_Data/Town02/data/*",
        "./PDM_Lite_Carla_LB2_Data/Town03/data/*/*",
    ])

    train_model(master_list)