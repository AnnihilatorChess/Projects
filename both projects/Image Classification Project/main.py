import os
import warnings
import numpy as np
import torch
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from architecture import MyCNN
from dataset import ImagesDataset
from utilities import plot_losses, TransformedImagesDataset, log_results


def train_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader, loss_fn, device, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data in tqdm(loader, desc="Training", position=0):
        images, labels, _, _ = data
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, accuracy


def evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels, _, _ = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            loss += loss_fn(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    loss /= len(loader)
    return loss, accuracy


def main(
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 16,
        data_set: str = "training_data"
):

    device = torch.device("cuda")
    if "cuda" in device.type and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    # Set a known random seed for reproducibility
    np.random.seed(1)
    torch.manual_seed(1)

    # datasets and dataloaders
    dataset = ImagesDataset(image_dir=data_set)

    total_samples = len(dataset)
    train_size = int(0.7 * total_samples)
    val_size = (total_samples - train_size)//2
    test_size = total_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    # Create augmented training data set and data loader
    training_set_augmented = TransformedImagesDataset(data_set=train_dataset)
    indices = np.arange(len(training_set_augmented))
    np.random.shuffle(indices)
    training_set_augmented_shuffled = torch.utils.data.Subset(training_set_augmented, indices)

    train_loader_augmented = torch.utils.data.DataLoader(
        training_set_augmented_shuffled,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )

    # Define network
    model = MyCNN()
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.875)

    # looping over data
    avg_loss_train = []
    avg_loss_val = []
    best_val_loss = np.inf
    torch.save(model.state_dict(), "model.pth")

    for i in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader_augmented, loss_fn, device, optimizer=optimizer)
        avg_loss_train.append(train_loss)
        val_loss, val_acc = evaluate_model(model=model, loader=val_loader, loss_fn=loss_fn, device=device)
        avg_loss_val.append(val_loss)
        scheduler.step()
        print("-" * 70)
        print(f"Epoch {i}:\n"
              f"tl {train_loss:7.2f} with acc {train_acc:7.2f}\n"
              f"vl {val_loss:7.2f}  with acc {val_acc:7.2f}\n")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "model.pth")
        if len(avg_loss_val) >= 10 and best_val_loss not in avg_loss_val[-7:]:
            num_epochs = i
            break

    print("Training done...")

    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    model.load_state_dict(torch.load(os.path.abspath("model.pth")))
    train_loss, train_acc = evaluate_model(model, loader=train_loader_augmented, loss_fn=loss_fn, device=device)
    val_loss, val_acc = evaluate_model(model, loader=val_loader, loss_fn=loss_fn, device=device)
    test_loss, test_acc = evaluate_model(model, loader=test_loader, loss_fn=loss_fn, device=device)

    print(f"Scores:")
    print(f"  training: loss {train_loss:7.3f} with acc {train_acc:7.2f}")
    print(f"validation: loss {val_loss:7.3f} with acc {val_acc:7.2f}")
    print(f"      test: loss {test_loss:7.3f} with acc {test_acc:7.2f}")

    # log results
    log_results(model, num_epochs, batch_size, optimizer, train_acc, val_acc, test_acc, val_loss, test_loss)
    # plot losses
    plot_losses(avg_loss_train, avg_loss_val, "epoch_loss")


if __name__ == "__main__":

    config = {
        "learning_rate": 0.005,
        "weight_decay": 0.002,
        "batch_size": 32,
        "data_set": "training_data_cleaned"
    }
    main(**config)
