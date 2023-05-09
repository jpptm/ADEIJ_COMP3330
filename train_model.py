import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from intel_dataloader import IntelDataLoader

from models.cv_model import CVModel

from tqdm import tqdm

# local modules
import metrics
import export

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


def test(model, val_loader, criterion, device):
    # Let model know we are in evaluation mode
    test_data = IntelTestLoader(csv_path)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    truth = []
    preds = []

    # Keep track of validation loss
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(
            test_loader,
            position=1,
            total=-len(test_loader),
            leave=False,
            desc="Testing"):

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

    export.Export(model, device, "Placeholder model name", history, val_loader)

    return avg_loss, acc


def main(data_path, lr, num_epochs, batch_size, loss, hidden_size, name, kind):
    # Set device - GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO]: USING {str(device).upper()} DEVICE")

    # Create dataset
    train_dataset = IntelDataLoader(data_path["train"])
    val_dataset = IntelDataLoader(data_path["val"])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model and optimiser
    model = CVModel(num_classes=6, hidden_size=hidden_size, kind=kind).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # History logging
    history = export.History()
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    # Train model
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch} of {num_epochs}")
        train_loss, train_acc = train(model, train_loader, loss, optimiser, device)
        val_loss, val_acc = validate(model, val_loader, loss, device)

        print(
            f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}%, "
            f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%"
        )

        # Save history
        history.append_all(train_loss, train_acc, val_loss, val_acc)

    # metrics.save_model(model)

    # proof of concept for metrics, can add into the inference also
    metrics.conf_matrix(model, val_loader, device)

    # Show loss and accuracy history
    plt.figure()
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()

    plt.figure()
    plt.plot(train_accs, label="Training accuracy")
    plt.plot(val_accs, label="Validation accuracy")
    plt.legend()
    # plt.show()
    test(data_path["test"], model, device, loss)


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(0)

    # Define hyperparameters
    data_paths = {
        "train": "./../ADEIJ_datasets/seg_train/seg_train",
        "val": "./../ADEIJ_datasets/seg_test/seg_test",
    }
    lr = 0.001
    num_epochs = 1
    batch_size = 32
    loss = torch.nn.CrossEntropyLoss()

    # for i in range(1, 6)
    input_map1 = {
        "data_path": data_paths,
        "lr": lr,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "loss": loss,
        "hidden_size": 30,
        "name": f"CVModel30_{i}",
        "kind": 'resnet50'
    }

    # input_map2 = {
    #     "data_path": data_paths,
    #     "lr": lr,
    #     "num_epochs": num_epochs,
    #     "batch_size": batch_size,
    #     "loss": loss,
    #     "hidden_size": 30
    #     "name": CVModel30
    # }

    # input_map2 = {
    #     "data_path": data_paths,
    #     "lr": lr,
    #     "num_epochs": num_epochs,
    #     "batch_size": batch_size,
    #     "loss": loss,
    #     "hidden_size": 30
    #     "name": CVModel30
    # }

    # Run main function
    main(**input_map1)
