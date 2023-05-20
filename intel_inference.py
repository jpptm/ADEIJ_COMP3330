import csv
import os

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


def inference(model_path, imgs_path):
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

    with open("preds.csv", "w", newline="") as f:
        writer = csv.writer(f)

        # Tqdm for progress bar
        for img, img_path in tqdm(inference_dataloader,
                                  total=len(inference_dataloader),
                                  desc="Inference"):

            predictions = model(img.to(device))

            writer.writerow([img_path[0], torch.argmax(predictions).item()])


if __name__ == "__main__":
    # NOTE The model path is expected to be in the project's root directory
    # NOTE The image path that contains the test images is expected to be in the before project's root directory
    # NOTE The file will be written in the project's root directory
    model_path = f"{os.path.join(os.getcwd(), 'intel_model.pt')}"

    folder_name = "ADEIJ_datasets"
    imgs_path = os.path.join(os.getcwd(), "..", folder_name, "seg_pred", "seg_pred")

    inference(model_path=model_path, imgs_path=imgs_path)
