# seg_train - training data with ~14000 entries
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
# NOTE: PyTorch wants images in (Channels, H, W) format, so we need to permute the image
class IntelDataLoader(torch.utils.data.Dataset):
    def __init__(self, data_path, affine_probability=0.3):
        self.affine_probability = affine_probability

        # Put all the classes that we know into a tuple
        self.known_classes = (
            "buildings",
            "forest",
            "glacier",
            "mountain",
            "sea",
            "street",
        )

        # Go through each subdirectory, generate a label for each entry and add it to the master data
        self.master_data = []

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

        # Remove this when training, the data loader should be the one to handle all the shuffling
        np.random.shuffle(self.master_data)

        # Data augmentations to be applied
        self.transforms = tv.transforms.Compose(
            [
                tv.transforms.ColorJitter(
                    brightness=(0.375, 1.2),
                    contrast=(0.3, 1.5),
                    saturation=(0.4, 1.3),
                    hue=(-0.1, 0.1),
                ),
                # I think with random cropping the images look blurred enough already but we can uncomment it any time if we need it
                # tv.transforms.GaussianBlur(3, sigma=(0.5, 1.5)),
                tv.transforms.RandomPosterize(bits=5, p=0.4),  # Bits to keep
                tv.transforms.RandomAdjustSharpness(sharpness_factor=1.5),
                tv.transforms.RandomEqualize(p=0.1),
                # Have to be real careful with how we include the random crop as it could confuse the network
                # IMO, we just don't use random crop cause it's very easy to confuse the network since the dataset has buildings and streets
                tv.transforms.RandomResizedCrop(size=(150, 150)),
                tv.transforms.RandomVerticalFlip(p=0.15),
                tv.transforms.RandomHorizontalFlip(p=0.5),
                tv.transforms.RandomPerspective(distortion_scale=0.5, p=0.3),
            ]
        )

    def __len__(self):
        return len(self.master_data)

    def __getitem__(self, idx):
        # Get the image name and label
        img_name, label = self.master_data[idx]

        # Read the image and convert it to a tensor
        img = cv2.imread(img_name)
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Convert the image to (Channels, H, W) format
        img = img.permute(2, 0, 1)

        # Apply transforms to image and normalise it
        img = self.transforms(img)

        # Apply random affine transforms - we do it this way because there is no p argument for this function
        # We don't necessarily want to rotate every image every single time
        if self.affine_probability > np.random.uniform():
            img = tv.transforms.RandomAffine(
                degrees=45, translate=(0.1, 0.1), shear=22.5
            )(img)
            print(True)

        # Comment the line/s below if you want to show the augmented images
        # img = img.type(torch.float32) / 255

        # Convert the label to a tensor
        label = torch.tensor(label)

        return img, label


# Uncomment to show the images
dl = IntelDataLoader("C:/Microsoft VS Code/ADEIJ_datasets/seg_train/seg_train")
for i in range(100):
    ci = dl.__getitem__(i)

    print(f"Label: {dl.known_classes[ci[1].int()]}")

    img = cv2.imshow(
        "", cv2.resize(torch.permute(ci[0], (1, 2, 0)).numpy(), (150, 150))
    )

    if cv2.waitKey(0) == 27:
        exit(0)


# Ideas on how to use (The dataloader should be able to shuffle it by itself using the shuffle argument)
# path = "C:/Microsoft VS Code/ADEIJ_datasets/seg_train/seg_train"
# s = IntelDataLoader(path)
# print(torch.Tensor.size(s.__get_item__(0)[0].permute(2, 0, 1)))
# dataset = IntelDataLoader(seg_train_path)
# train/test_data = DataLoader(dataset, batch_size=32, shuffle=True)
# for i, data in enumerate(train_data): ...
