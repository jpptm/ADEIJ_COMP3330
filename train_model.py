import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from intel_dataloader import IntelDataLoader, IntelTestLoader

from models.cv_model import CVModel

from tqdm import tqdm

import math

# local modules
import metrics
import export
import pdb


# Add training function
def train(model, train_loader, criterion, optimiser, device):
    # Let model know we are in training mode
    model.train()

    # Keep track of training loss and accuracy
    train_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(
        train_loader,
        position=1,
        total=len(train_loader),
        leave=False,
        desc="Training",
    ):
        # Cast tensors to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Reset gradients
        optimiser.zero_grad()

        # Get model outputs and calculate loss
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backpropagate and update optimiser learning rate
        loss.backward()
        optimiser.step()

        # Keep track of loss and accuracy
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    avg_loss = train_loss / len(train_loader)

    return avg_loss, acc


def validate(model, val_loader, criterion, device):
    # Let model know we are in evaluation mode
    model.eval()

    # Keep track of validation loss
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader,
                                    position=1,
                                    total=len(val_loader),
                                    leave=False,
                                    desc="Validating"):
            # Cast tensors to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Calculate model output and loss
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Keep track of loss and accuracy
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    avg_loss = val_loss / len(val_loader)

    return avg_loss, acc


def test(csv_path, model, device, criterion, history, name, epoch):

    test_data = IntelTestLoader(csv_path)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    truth = []
    preds = []

    # Keep track of validation loss
    val_loss = 0
    correct = 0
    total = 0

    # confusion_mat = confusion_matrix(truth, preds)
    # acc = accuracy_score(truth, preds)

    # precision_global = precision_score(truth, preds, average="micro")
    # precision_mean = precision_score(truth, preds, average="macro")

    # recall_global = recall_score(truth, preds, average="micro")
    # recall_mean = recall_score(truth, preds, average="macro")

    # avg_loss = val_loss / len(val_loader)

    export.Export(model, device, name, history, test_loader, epoch)


def main(data_path, lr, max_epochs, batch_size, loss, hidden_size, name, kind, test_every, use_learning_decay):
    # Set device - GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]: USING {str(device).upper()} DEVICE")

    print("Parallel test!")

    # Create dataset
    train_dataset = IntelDataLoader(data_path["train"])
    val_dataset = IntelDataLoader(data_path["val"])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model and optimiser
    model = CVModel(num_classes=6, hidden_size=hidden_size, kind=kind).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    if use_learning_decay:
        scheduler = CosineAnnealingLR(optimiser, T_max=max_epochs, eta_min=0)

    # History logging
    history = export.History()
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    # Train model
    for epoch in range(1, max_epochs + 1):
        print(f"Epoch {epoch} of {max_epochs}")
        train_loss, train_acc = train(model, train_loader, loss, optimiser, device)
        if use_learning_decay:
            # Update learning rate according to cosine annealing
            scheduler.step()
        val_loss, val_acc = validate(model, val_loader, loss, device)

        print(
            f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}%, "
            f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%"
        )

        # Save history
        history.append_all(train_loss, train_acc, val_loss, val_acc)

        if epoch%test_every == 0 or epoch == max_epochs:
            save_name = name + "_epochs_" + str(epoch)
            test(data_path["test_csv"], model, device, loss, history, save_name, epoch)

    # metrics.save_model(model)

    # proof of concept for metrics, can add into the inference also
    # metrics.conf_matrix(model, val_loader, device)

    # Show loss and accuracy history
    # plt.figure()
    # plt.plot(train_losses, label="Training loss")
    # plt.plot(val_losses, label="Validation loss")
    # plt.legend()

    # plt.figure()
    # plt.plot(train_accs, label="Training accuracy")
    # plt.plot(val_accs, label="Validation accuracy")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(0)

    # Define hyperparameters
    data_paths = {
        "train": "./../ADEIJ_datasets/seg_train/seg_train",
        # "train": "./../ADEIJ_datasets/pretend_train",
        "val": "./../ADEIJ_datasets/seg_test/seg_test",
        "test_csv": "./../ADEIJ_datasets/seg_pred_labels.csv"
    }
    lr = 0.0001

    # epochs settings
    max_epochs = 1
    test_every = 2

    # training settings
    batch_size = 64
    loss = torch.nn.CrossEntropyLoss()
    use_learning_decay = False

    # model settings
    kind = 'vit'
    hidden_size = 90
    hidden_size_increment = 10

    for i in range(1, 3):
        if i == 1:
            hidden_size = 150
        if i == 1:
            hidden_size = 20
        elif i == 2:
            hidden_size = 30
        elif i == 3:
            hidden_size = 150
        
        input_map = {
            "data_path": data_paths,
            "lr": lr,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "loss": loss,
            "hidden_size": hidden_size,
            "name": f"{kind}_{hidden_size}",
            "kind": kind,
            "test_every": test_every,
            "use_learning_decay": use_learning_decay
        }

        hidden_size += hidden_size_increment

        # Run main function
        main(**input_map)

    # input_map2 = {
    #     "data_path": data_paths,
    #     "lr": lr,
    #     "max_epochs": max_epochs,
    #     "batch_size": batch_size,
    #     "loss": loss,
    #     "hidden_size": 30
    #     "name": CVModel30
    # }

    # input_map2 = {
    #     "data_path": data_paths,
    #     "lr": lr,
    #     "max_epochs": max_epochs,
    #     "batch_size": batch_size,
    #     "loss": loss,
    #     "hidden_size": 30
    #     "name": CVModel30
    # }
