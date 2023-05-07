import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm
import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    # normalize
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# can use this at the end of train_model


def conf_matrix(model, data_loader, device):
    # Get predictions for the test set
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader,
                                    position=1,
                                    total=len(data_loader),
                                    leave=False,
                                    desc="Testing",
                                    ):
            # Cast tensors to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Get model predictions
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Save predictions and labels
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(targets.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(test_labels, test_preds)

    classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

    # for later plotting
    return (cm, classes)
