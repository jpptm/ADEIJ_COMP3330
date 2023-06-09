import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

# local modules
import metrics


class Export:
    intel_classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    preds, labels = None, None

    def __init__(self, model, device, history, loader, base_path='./outputs/'):
        self.model = model
        self.device = device
        self.name = model.name
        self.history = history

        # path to store stuff in
        self.path = base_path + self.name + '/'

        # Create a folder to store the model info in (default is ./outputs/<name>/)
        # If the directory already exists then nothing will happen
        os.makedirs(self.path, exist_ok=True)

        # guess what this one does
        self.save_model_to_disk()

        # make predictions if a loader was passed
        self.preds, self.labels = self.predict(loader)

        # basic plots
        if self.history is not None:
            self.loss_acc_plots(save_to_file=True)

        if self.preds is not None and self.labels is not None:
            self.cm_plot(confusion_matrix(self.preds, self.labels), self.intel_classes, save_to_file=True)

        plt.close('all')

    # make predictions based on given dataloader
    def predict(self, loader):
        preds = []
        labels = []

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in tqdm(loader,
                                        position=1,
                                        total=len(loader),
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

                # get alternative accuracy values
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            acc = 100.0 * correct / total

            # global is avg across all guesses
            # mean is avg of all classes
            # if that makes sense lol
            precision_global = precision_score(labels, preds, average="micro")
            precision_mean = precision_score(labels, preds, average="macro")

            recall_global = recall_score(labels, preds, average="micro")
            recall_mean = recall_score(labels, preds, average="macro")

        self.save_stats(acc, precision_global, precision_mean, recall_global, recall_mean)

        return preds, labels


    def save_model_to_disk(self):
        model_path = self.path + self.name + '_model.pt'
        torch.save(self.model.state_dict(), model_path)


    # save the final tran loss and accuracy and stuff and things
    def save_stats(self, acc, precision_global, precision_mean, recall_global, recall_mean):
        try:
            stats_path = self.path + self.name + '_stats.txt'
            with open(stats_path, 'a') as f:
                if self.history is not None:
                    f.write(
                        f"Train Loss = {self.history.train_losses[-1]:.4f}, Train Acc = {self.history.train_accs[-1]:.2f}%, "
                        f"Val Loss = {self.history.val_losses[-1]:.4f}, Val Acc = {self.history.val_accs[-1]:.2f}%\n, "
                    )
                f.write(
                    f"Final Test Accuracy = {acc}%\n, "
                    f"Precision Global = {precision_global}%\n, "
                    f"Precision Mean = {precision_mean}%\n, "
                    f"Recall Global = {recall_global}%\n, "
                    f"Recall Mean = {recall_mean}%\n, "
                )
        except Exception as e:
            print(f"Error saving stats to file: {str(e)}")

    # old code, ignore
    # def lazy_plot(self):
    #     # Show loss and accuracy history
    #     plt_loss = plt.figure(1)
    #     plt.plot(self.history.train_losses, label="Training loss")
    #     plt.plot(self.history.val_losses, label="Validation loss")
    #     plt.legend()

    #     plt_acc = plt.figure(2)
    #     plt.plot(self.history.train_accs, label="Training accuracy")
    #     plt.plot(self.history.val_accs, label="Validation accuracy")
    #     plt.legend()

    #     plt_cm = plt.figure(3)
    #     # Compute confusion matrix
    #     cm = confusion_matrix(self.labels, self.preds)
    #     classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    #     metrics.plot_confusion_matrix(cm, classes)

        # show all figures
        # plt.show()

    def loss_acc_plots(self, save_to_file=True, show_plot=False):
        # Generate plot for loss
        loss_fig, loss_ax = plt.subplots()
        loss_ax.plot(self.history.train_losses, label="Training loss")
        loss_ax.plot(self.history.val_losses, label="Validation loss")
        loss_ax.legend()

        # Generate plot for accuracy
        acc_fig, acc_ax = plt.subplots()
        acc_ax.plot(self.history.train_accs, label="Training accuracy")
        acc_ax.plot(self.history.val_accs, label="Validation accuracy")
        acc_ax.legend()

        # Save the plots if needed
        if save_to_file:
            # bbox_inches cuts whitespace
            loss_fig.savefig(self.path + 'loss.png', bbox_inches='tight')
            acc_fig.savefig(self.path + 'accuracy.png', bbox_inches='tight')

        # and show them if needed
        if show_plot:
            loss_fig.show()
            acc_fig.show()

    def cm_plot(self, cm, classes, save_to_file=True, show_plot=False, cmap=plt.cm.Blues):
        # normalise the confusion matrix
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # generate axis
        fig, ax = plt.subplots()

        # display matrix as an image
        cm_im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        fig.colorbar(cm_im, ax=ax)

        # put classes on the axes
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks, classes, rotation=45)
        ax.set_yticks(tick_marks, classes)

        # add text with correct colour
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black")

        # Add labels and title
        ax.set_title('Confusion matrix')
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        fig.tight_layout()

        # save and show the image according to passed parameters
        if save_to_file:
            fig.savefig(self.path + 'matrix.png', bbox_inches='tight')
        if show_plot:
            fig.show()


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
