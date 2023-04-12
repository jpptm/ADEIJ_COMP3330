# seg_train - training data with 14000 entries
#  seg_test - validation data with 3000 entries
#  seg_pred - unlabeled data with 7000 entries (test set used in the competition)
# mock_test - labelled test set with 12 images

import os

import cv2
import numpy as np
import torch
import torchvision as tv


# Assumes that the given path contains all the folders with all the data inside it
# NOTE: We take the image names and process this instead of loading everything at once into memory. This way, we make sure we don't exhaust our memory and crash the program
# NOTE: This class should work from both training and validation sets, as long as the right path is given
class IntelDataLoader(torch.utils.data.Dataset):
    def __init__(self, data_path):
        # Put all the classes that we know into a tuple
        self.known_classes = (
            "buildings",
            "forest",
            "glacier",
            "mountain",
            "sea",
            "street",
        )

        # Gaussian Blur
        # Random Invert?
        # Random Posterize
        # Random Adjust Sharpness
        # Random Autocontrast
        # Random Equalize
        # Random Solarize?

        # Random crop (Not quite sure about this one)
        # Vertical flip (Not quite sure about this one, it could be detrimental to the network's performance)
        # Horizontal flip
        # Random Perspective
        # Random affine

        # Normalise

        # Data augmentations to be applied
        transforms = []

        self.master_data = []

        # Go through each subdirectory, generate a label for each entry and add it to the master data
        for idx, member in enumerate(self.known_classes):
            current_path = os.path.join(data_path, member)
            full_names = [
                os.path.join(current_path, img) for img in os.listdir(current_path)
            ]

            no_imgs = len(os.listdir(current_path))
            labels = np.ones(no_imgs) * idx

            # Bind labels and image names together then add each to the master data list
            for item in zip(full_names, labels):
                self.master_data.append(item)

    def __len__(self):
        return len(self.master_data)

    def __get_item__(self, idx):
        # Get the image name and label
        img_name, label = self.master_data[idx]

        # Read the image and convert it to a tensor
        img = cv2.imread(img_name)
        img = torch.from_numpy(img)
        print(img)
        # Convert the label to a tensor
        label = torch.tensor(label)

        return img, label


# Ideas on how to use
# path = "C:/Microsoft VS Code/ADEIJ_datasets/seg_train/seg_train"
# s = IntelDataLoader(path)
# print(torch.Tensor.size(s.__get_item__(0)[0].permute(2, 0, 1)))
# dataset = IntelDataLoader(seg_train_path)
# DataLoader(dataset, batch_size=32, shuffle=True)
