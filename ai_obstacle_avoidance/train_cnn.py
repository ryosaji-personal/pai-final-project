import re
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


# ==========================================
# 設定
# ==========================================

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "obstacle_cnn_best.pth"

CLASSES = ["none", "left", "center", "right"]

WIDTH = 160
HEIGHT = 120

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 7
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# データセット
# ==========================================

class RobotDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        image = cv2.imread(str(path))

        if image is None:
            raise RuntimeError(f"Cannot read: {path}")

        image = cv2.resize(
            image,
            (WIDTH, HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = np.ascontiguousarray(
            image.transpose(2, 0, 1)
        )

        image = torch.from_numpy(image).float() / 255.0

        return image, label


# ==========================================
# データ読み込み
# ==========================================

def load_samples():

    samples = []
    ignored = []

    for label, name in enumerate(CLASSES):

        directory = DATASET_DIR / name

        for path in sorted(directory.glob("*.png")):

            match = re.match(
                r"^pos(\d+)_img\d+\.png$",
                path.name
            )

            if match is None:
                ignored.append(path)
                continue

            position_id = int(match.group(1))

            samples.append(
                (path, label, position_id)
            )

    print(f"Loaded: {len(samples)} images")

    if ignored:
        print(
            f"Ignored {len(ignored)} images "
            "without position IDs"
        )

    return samples


def split_by_position(samples):

    positions = sorted(set(s[2] for s in samples))

    if len(positions) < 2:
        raise RuntimeError(
            "At least two recording positions are required"
        )

    # 最後の撮影地点を優先する。
    # 両方の集合に全クラスが存在する地点を選ぶ。
    candidates = []

    for pos in positions:

        train = [s for s in samples if s[2] != pos]
        val = [s for s in samples if s[2] == pos]

        train_labels = {s[1] for s in train}
        val_labels = {s[1] for s in val}

        if (
            len(train_labels) == len(CLASSES)
            and len(val_labels) == len(CLASSES)
        ):
            candidates.append(pos)

    if not candidates:
        raise RuntimeError(
            "No single recording position contains all "
            "classes while leaving all classes in training. "
            "Review the data distribution."
        )

    val_position = max(candidates)

    train_samples = [
        (s[0], s[1])
        for s in samples
        if s[2] != val_position
    ]

    val_samples = [
        (s[0], s[1])
        for s in samples
        if s[2] == val_position
    ]

    print(f"Validation position: pos{val_position:02d}")
    print(f"Training:   {len(train_samples)}")
    print(f"Validation: {len(val_samples)}")

    return train_samples, val_samples, val_position


# ==========================================
# CNN
# ==========================================

class ObstacleCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(CLASSES))
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# ==========================================
# 評価
# ==========================================

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    total = 0

    confusion = np.zeros(
        (len(CLASSES), len(CLASSES)),
        dtype=np.int64
    )

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total += batch_size

            predictions = outputs.argmax(dim=1)

            for truth, pred in zip(
                labels.cpu().tolist(),
                predictions.cpu().tolist()
            ):
                confusion[truth, pred] += 1

    return total_loss / total, confusion


def print_metrics(confusion):

    print("\n===== Confusion Matrix =====")
    print("Rows: True labels")
    print("Cols: Predicted labels")
    print("       none  left  center right")

    for name, row in zip(CLASSES, confusion):
        print(f"{name:7s}", *[f"{n:6d}" for n in row])

    print("\n===== Classification Report =====")
    print(
        f"{'Class':8s} "
        f"{'Precision':>10s} "
        f"{'Recall':>10s} "
        f"{'F1':>10s} "
        f"{'Support':>8s}"
    )

    recalls = []

    for i, name in enumerate(CLASSES):

        tp = confusion[i, i]

        support = confusion[i, :].sum()
        predicted = confusion[:, i].sum()

        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        recalls.append(recall)

        print(
            f"{name:8s} "
            f"{precision:10.3f} "
            f"{recall:10.3f} "
            f"{f1:10.3f} "
            f"{support:8d}"
        )

    accuracy = (
        np.trace(confusion) / confusion.sum()
        if confusion.sum() else 0.0
    )

    print(f"\nAccuracy:     {accuracy:.4f}")
    print(f"Macro Recall: {np.mean(recalls):.4f}")


# ==========================================
# 学習
# ==========================================

def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_samples()

    train_samples, val_samples, val_position = (
        split_by_position(samples)
    )

    train_counts = np.bincount(
        [label for _, label in train_samples],
        minlength=len(CLASSES)
    )

    val_counts = np.bincount(
        [label for _, label in val_samples],
        minlength=len(CLASSES)
    )

    print("\nClass distribution:")
    for i, name in enumerate(CLASSES):
        print(
            f"{name:8s} "
            f"train={train_counts[i]:4d} "
            f"val={val_counts[i]:4d}"
        )

    # クラス数が少ないほど損失に大きな重みを与える
    weights = (
        train_counts.sum()
        / (len(CLASSES) * train_counts)
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device
    )

    print("\nClass weights:", class_weights.tolist())

    train_loader = DataLoader(
        RobotDataset(train_samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        RobotDataset(val_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = ObstacleCNN().to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    no_improvement = 0

    # --------------------------------------
    # 学習ループ
    # --------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_train_loss = 0.0
        total_train = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)

            total_train_loss += (
                loss.item() * batch_size
            )
            total_train += batch_size

        train_loss = total_train_loss / total_train

        val_loss, confusion = evaluate(
            model,
            val_loader,
            criterion
        )

        val_accuracy = (
            np.trace(confusion) / confusion.sum()
        )

        center_recall = (
            confusion[2, 2] / confusion[2].sum()
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"TrainLoss={train_loss:.4f} "
            f"ValLoss={val_loss:.4f} "
            f"ValAcc={val_accuracy:.4f} "
            f"CenterRecall={center_recall:.4f}"
        )

        # 最良モデルを保存
        if val_loss < best_val_loss - 1e-5:

            best_val_loss = val_loss
            no_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "width": WIDTH,
                    "height": HEIGHT,
                    "validation_position": val_position,
                },
                MODEL_PATH
            )

            print("  -> Best model saved")

        else:

            no_improvement += 1

            if no_improvement >= PATIENCE:
                print("Early stopping")
                break

    # --------------------------------------
    # 最良モデルを再読み込みして評価
    # --------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    final_loss, confusion = evaluate(
        model,
        val_loader,
        criterion
    )

    print("\n===== Final Evaluation =====")
    print(f"Validation loss: {final_loss:.4f}")

    print_metrics(confusion)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
