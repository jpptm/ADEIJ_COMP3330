import csv
import os
import argparse

import cv2
import torch
from tqdm import tqdm

from models.cv_model import CVModel


# Class to load image to a data loader so we can batch our inferences
class InferenceLoader(torch.utils.data.Dataset):
    def __init__(self, imgs_path):
        # If file in directory is a .jpg add it to the list
        self.master_list = [os.path.join(imgs_path, f)
                            for f in os.listdir(imgs_path)
                            if f.endswith(".jpg")]

    def __len__(self):
        return len(self.master_list)

    def __getitem__(self, idx):
        # Load image and permute image to (C, H, W) from (H, W, C) after transforming from BGR to RGB
        # The colour change is because cv2 loads images in the BGR format for some reason
        img_path = self.master_list[idx]
        img = cv2.imread(img_path)
        # Make sure no images have the wrong shape
        img = (cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (150, 150))
               if img.shape != (150, 150, 3)
               else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1), img_path


def inference(model_path, imgs_path, out_path):
    # Load model
    model = CVModel(num_classes=6)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load data
    inference_data = InferenceLoader(imgs_path)
    inference_dataloader = torch.utils.data.DataLoader(inference_data, shuffle=False)

    # Load known classes
    classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Tqdm for progress bar
        for img, img_path in tqdm(inference_dataloader,
                                  total=len(inference_dataloader),
                                  desc="Inference"):

            predictions = model(img.to(device))

            writer.writerow([img_path[0], torch.argmax(predictions).item()])


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add the model argument
    parser.add_argument('-m', '--model', type=str, help='Specify the model path')
    # Add the img-folder argument
    parser.add_argument('-i', '--image-folder', type=str, default='../ADEIJ_datasets/seg_pred/seg_pred', help='Specify the image folder path')
    # Add the output argument (optional)
    parser.add_argument('-o', '--output', type=str, default='preds.csv', help='Specify the output CSV file name (optional)')

    # Parse the command-line arguments
    args = parser.parse_args()

    inference(model_path=args.model, imgs_path=args.image_folder, out_path=args.output)
