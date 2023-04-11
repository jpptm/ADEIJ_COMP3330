# seg_train - training data with 14000 entries
#  seg_test - validation data with 3000 entries
#  seg_pred - unlabeled data with 7000 entries (test set used in the competition)
# mock_test - labelled test set with 12 images

import os

import numpy as np
import torch
import cv2

# Assumes that the given path contains all the folders with all the data inside it
class IntelDataLoader(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.known_classes = (
            "buildings",
            "forest",
            "glacier",
            "mountain",
            "sea",
            "street",
        )

        self.master_data = []
        for idx, member in enumerate(self.known_classes):
            current_path = os.path.join(data_path, member)
            no_imgs = len(os.listdir(current_path))

            full_names = [
                os.path.join(current_path, img) for img in os.listdir(current_path)
            ]

            # cv2.imshow("image", cv2.resize(cv2.imread(full_names[100]), (450, 450)))
            # cv2.waitKey(0)
            # print(full_names)
            # exit(0)

            labels = np.ones(no_imgs) * idx

            for item in zip(full_names, labels):
                self.master_data.append(item)

        print(len(self.master_data))

        for data in self.master_data:
            print(data)
            input("...")

    def __len__(self):
        return len(self.master_data)

    def __get_item__(self):
        pass


path = os.path.join(os.getcwd(), "..", "ADEIJ_datasets", "seg_test", "seg_test")
IntelDataLoader(path)
