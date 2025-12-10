import os
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

data_root = r"../data"

img_size = 64  # resize all images to 64x64 for the MLP
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

train_dir = os.path.join(data_root, "train")
test_dir = os.path.join(data_root, "test")

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

print("Classes:", train_dataset.classes)  # expected ['fresh', 'rotten']


# Train/validation split
val_fraction = 0.2
val_size = int(len(train_dataset) * val_fraction)
train_size = len(train_dataset) - val_size

train_subset, val_subset = random_split(
    train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

def make_loaders(batch_size: int):
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,   
        pin_memory=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader

# Infer input dim & classes
x0, _ = train_dataset[0]
input_dim = x0.numel()  # C * H * W
num_classes = len(train_dataset.classes)
print("Input dim:", input_dim, "| Num classes:", num_classes)

# assume classes are ['fresh', 'rotten'] and treat 'rotten' as positive
FRESH_IDX = train_dataset.class_to_idx['fresh']
ROTTEN_IDX = train_dataset.class_to_idx['rotten']

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten to (batch_size, C*H*W)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Evaluation function
def evaluate(model: nn.Module, data_loader: DataLoader):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * labels.size(0)

            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_loss = loss_sum / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0

    # Compute binary F1 with 'rotten' as positive
    if total > 0:
        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_labels)

        tp = ((y_pred == ROTTEN_IDX) & (y_true == ROTTEN_IDX)).sum().item()
        fp = ((y_pred == ROTTEN_IDX) & (y_true == FRESH_IDX)).sum().item()
        fn = ((y_pred == FRESH_IDX) & (y_true == ROTTEN_IDX)).sum().item()

        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom > 0 else 0.0
    else:
        f1 = 0.0

    return avg_loss, acc, f1

# Train one hyperparam setting
def train_one_setting(hidden_dim: int, lr: float, weight_decay: float,
                      batch_size: int, num_epochs: int = 30):
    print(f"\nTraining with: hidden={hidden_dim}, lr={lr}, wd={weight_decay}, batch={batch_size}")
    train_loader, val_loader, _ = make_loaders(batch_size)

    model = MLP(input_dim, hidden_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = 0.0
    best_state_dict = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0

        val_loss, val_acc, val_f1 = evaluate(model, val_loader)

        # Track best val F1 for the current config
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = model.state_dict()

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f}, acc {train_acc*100:5.2f}% | "
            f"val loss {val_loss:.4f}, acc {val_acc*100:5.2f}%, F1 {val_f1:.4f}"
        )

    return best_val_f1, best_state_dict

# Hyperparameter grid search
if __name__ == "__main__":
    hidden_dims = [128, 256, 512]
    learning_rates = [1e-3, 3e-4, 1e-4]
    weight_decays = [0.0, 1e-4, 1e-3]
    batch_sizes = [16, 32, 64]

    # Cartesian product of all hyperparameters
    search_space = list(itertools.product(hidden_dims, learning_rates, weight_decays, batch_sizes))
    print(f"Number of configs to try: {len(search_space)}") 

    best_config = None
    best_f1 = 0.0
    best_state_dict = None

    for i, (hidden_dim, lr, wd, bs) in enumerate(search_space, start=1):
        print("\n" + "#" * 60)
        print(
            f"Config {i}/{len(search_space)}: "
            f"hidden={hidden_dim}, lr={lr}, wd={wd}, batch={bs}"
        )

        val_f1, state_dict = train_one_setting(
            hidden_dim=hidden_dim,
            lr=lr,
            weight_decay=wd,
            batch_size=bs,
            num_epochs=30 
        )
        print(f"Validation F1 for this config: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_config = {
                "hidden_dim": hidden_dim,
                "lr": lr,
                "weight_decay": wd,
                "batch_size": bs,
            }
            best_state_dict = state_dict

    print("Best config:", best_config)
    print(f"Best validation F1: {best_f1:.4f}")

    # Save best weights
    if best_state_dict is not None and best_config is not None:
        weights_path = "fruit_mlp_best.pt"  # <- .pt file
        torch.save(best_state_dict, weights_path)
        print(f"Saved best model weights to {weights_path}")

        print("\nBest model evaluation on test set:")
        _, _, test_loader = make_loaders(best_config["batch_size"])
        best_model = MLP(input_dim, best_config["hidden_dim"], num_classes).to(device)
        best_model.load_state_dict(best_state_dict)
        test_loss, test_acc, test_f1 = evaluate(best_model, test_loader)
        print(
            f"Test loss: {test_loss:.4f}, "
            f"test acc: {test_acc*100:.2f}%, "
            f"test F1: {test_f1:.4f}"
        )
    else:
        print("something went wrong")
