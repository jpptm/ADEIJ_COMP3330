import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# local modules
import metrics


class Export:
    def __init__(self, model, device, name, history, val_loader=None, base_path='./outputs/'):
        self.model = model
        self.device = device
        self.name = name
        self.history = history

        # path to store stuff in
        self.path = base_path + name + '/'

        # Create a folder to store the model info in (default is ./outputs/<name>/)
        # If the directory already exists then nothing will happen
        os.makedirs(self.path, exist_ok=True)

        # predictions for the validation set if a loader was provided
        if val_loader is not None:
            self.val_preds, self.val_labels = self.predict(val_loader)

        # just do the plots for now
        self.lazy_plot()

    # make predictions on the test set
    def predict(self, val_loader):
        preds = []
        labels = []
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader,
                                        position=1,
                                        total=len(val_loader),
                                        leave=False,
                                        desc="Testing "+self.name,
                                        ):
                # Cast tensors to device
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                # Get model predictions
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # Save predictions and labels
                preds.extend(predicted.cpu().numpy())
                labels.extend(targets.cpu().numpy())

        return preds, labels

    def save_model_to_disk(self):
        model_path = self.path + self.name + '_model.pt'
        torch.save(self.model.state_dict(), model_path)

    # copied directly over
    def lazy_plot(self):
        # Show loss and accuracy history
        plt_loss = plt.figure(1)
        plt.plot(self.history.train_losses, label="Training loss")
        plt.plot(self.history.val_losses, label="Validation loss")
        plt.legend()

        plt_acc = plt.figure(2)
        plt.plot(self.history.train_accs, label="Training accuracy")
        plt.plot(self.history.val_accs, label="Validation accuracy")
        plt.legend()

        plt_cm = plt.figure(3)
        # Compute confusion matrix
        cm = confusion_matrix(self.val_labels, self.val_preds)
        classes = ["buildings", "forest",
                   "glacier", "mountain", "sea", "street"]
        metrics.plot_confusion_matrix(cm, classes)

        # show all figures
        plt.show()


class History:  # Standard format for model history
    def __init__(self):
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

    # to make appending concise
    def append_all(self, train_loss, train_acc, val_loss, val_acc):
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
