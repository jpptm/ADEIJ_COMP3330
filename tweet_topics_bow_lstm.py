import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from tweet_topics_dataloader import TweetTopicsDataLoader
from tweet_topics_models import LSTM, BoW


def train(model, dataloader, loss_fn, optimizer, device, column_name):
    model.train()
    losses, accuracies = [], []
    for batch in tqdm(
        dataloader, position=1, total=len(dataloader), leave=False, desc="Training"
    ):
        inputs = batch[column_name].to(device)
        labels = batch["label"].to(device)
        # Reset the gradients for all variables
        optimizer.zero_grad()
        # Forward pass
        preds = model(inputs)
        # Calculate loss
        loss = loss_fn(preds, labels)
        # Backward pass
        loss.backward()
        # Adjust weights
        optimizer.step()
        # Log
        losses.append(loss.detach().cpu().numpy())
        accuracy = torch.sum(torch.argmax(preds, dim=-1) == labels) / labels.shape[0]
        accuracies.append(accuracy.detach().cpu().numpy())

    return np.mean(losses), np.mean(accuracies)


def val(model, dataloader, loss_fn, device, column_name):
    model.eval()
    losses, accuracies = [], []
    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            position=1,
            total=len(dataloader),
            leave=False,
            desc="Validating",
        ):
            inputs = batch[column_name].to(device)
            labels = batch["label"].to(device)
            # Forward pass
            preds = model(inputs)
            # Calculate loss
            loss = loss_fn(preds, labels)
            # Log
            losses.append(loss.detach().cpu().numpy())
            accuracy = (
                torch.sum(torch.argmax(preds, dim=-1) == labels) / labels.shape[0]
            )
            accuracies.append(accuracy.detach().cpu().numpy())

    return np.mean(losses), np.mean(accuracies)


# Test on the test dataset
def test(dataloader, model, device, criterion, column_name):

    truth = []
    preds = []

    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            position=1,
            total=len(dataloader),
            leave=False,
            desc="Testing",
        ):
            # Cast tensors to device
            inputs = batch[column_name].to(device)
            targets = batch["label"].to(device)

            # Calculate model output and loss
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Keep track of loss and accuracy
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            truth.append(targets.cpu().numpy().flatten())
            preds.append(predicted.cpu().numpy().flatten())

    # Flatten outputs
    truth = [item for sublist in truth for item in sublist]
    preds = [item for sublist in preds for item in sublist]

    # Get accuracy, precision and recall
    confusion_mat = confusion_matrix(truth, preds)
    acc = accuracy_score(truth, preds)

    precision_global = precision_score(truth, preds, average="micro")
    precision_mean = precision_score(truth, preds, average="macro")

    recall_global = recall_score(truth, preds, average="micro")
    recall_mean = recall_score(truth, preds, average="macro")

    print("Confusion matrix: \n{}\n".format(confusion_mat))
    print(" Accuracy - {:.4f}".format(acc))
    print(
        "Precision - Global: {:.4f} \t Mean: {:.4f}".format(
            precision_global, precision_mean
        )
    )
    print(
        "   Recall - Global: {:.4f} \t Mean: {:.4f}".format(recall_global, recall_mean)
    )


def main(
    optimiser,
    device,
    loss,
    learning_rate,
    batch_size,
    hidden_layers,
    activation,
    n_embed,
    num_epochs,
    column_name,
    lstm_hidden,
):

    # For BoW
    master_data = TweetTopicsDataLoader(BoW=True, batch_size=batch_size)
    model = BoW(len(master_data.vocab), hidden_layers, activation)

    # For LSTM
    # master_data = TweetTopicsDataLoader(BoW=False, batch_size=batch_size)
    # model = LSTM(
    #     len(master_data.vocab),
    #     n_embed=n_embed,
    #     pad_index=master_data.pad_index,
    #     hidden_size=lstm_hidden,
    # )

    optimiser = optimiser(model.parameters(), lr=learning_rate)

    # Load data
    train_dataloader = master_data.train
    val_dataloader = master_data.val
    test_dataloader = master_data.test

    # Logging
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    for epoch in range(num_epochs):
        # Train
        train_loss, train_accuracy = train(
            model, train_dataloader, loss, optimiser, device, column_name
        )
        # Evaluate
        valid_loss, valid_accuracy = val(
            model, val_dataloader, loss, device, column_name
        )
        # Log
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)
        val_losses.append(valid_loss)
        val_accs.append(valid_accuracy)
        print(
            "Epoch {}: train_loss={:.4f}, train_accuracy={:.4f}, valid_loss={:.4f}, valid_accuracy={:.4f}".format(
                epoch + 1, train_loss, train_accuracy, valid_loss, valid_accuracy
            )
        )

    # Test
    test(test_dataloader, model, device, loss, column_name)

    # Show loss and accuracy history
    plt.figure()
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()

    plt.figure()
    plt.plot(train_accs, label="Training accuracy")
    plt.plot(val_accs, label="Validation accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Debugging
    main(
        optimiser=torch.optim.Adam,
        device="cpu",
        loss=torch.nn.CrossEntropyLoss(),
        learning_rate=0.0001,
        batch_size=64,
        hidden_layers=[100, 100, 100],  # BoW hidden layers
        activation=torch.nn.ReLU(),  # BoW activations
        num_epochs=35,
        column_name="multi_hot",  # multi_hot for BoW, ids for LSTM
        n_embed=128,
        lstm_hidden=32,
    )
