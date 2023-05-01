import csv
import os

import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from models.cv_model import CVModel


# Class to load image to a data loader so we can batch our inferences
class InferenceLoader(torch.utils.data.Dataset):
    def __init__(self, imgs_path):
        # If file in directory is a .jpg add it to the list
        self.master_list = [
            os.path.join(imgs_path, f)
            for f in os.listdir(imgs_path)
            if f.endswith(".jpg")
        ]

    def __len__(self):
        return len(self.master_list)

    def __getitem__(self, idx):
        # Load image and permute image to (C, H, W) from (H, W, C) after transforming from BGR to RGB
        # The colour change is because cv2 loads images in the BGR format for some reason
        img_path = self.master_list[idx]
        img = cv2.imread(img_path)
        # Make sure no images have the wrong shape
        img = (
            cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (150, 150))
            if img.shape != (150, 150, 3)
            else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )

        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1), img_path


def inference(model_path, imgs_path, show=False):
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
    class_map = {k: v for k, v in zip(range(len(classes)), classes)}

    with open("preds.csv", "w", newline="") as f:
        writer = csv.writer(f)

        if show:
            batch_list = []

        # Tqdm for progress bar
        for img, img_path in tqdm(
            inference_dataloader,
            total=len(inference_dataloader),
            desc="Inference",
        ):

            predictions = model(img.to(device))
            writer.writerow([img_path[0], torch.argmax(predictions).item()])

            if show:
                batch = {"img_name": img_path, "preds": predictions}
                batch_list.append(batch)

                if len(batch_list) == 64:
                    # Load as images in a subplot
                    fig, axs = plt.subplots(nrows=8, ncols=8, figsize=(10, 10))
                    for i, batch in enumerate(batch_list):
                        img_name = batch["img_name"]
                        preds = batch["preds"]

                        img = cv2.imread(img_name[0])
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # Use the index to put the images into the correct subplot
                        axs[i // 8, i % 8].imshow(img)
                        axs[i // 8, i % 8].set_title(
                            f"Prediction: {class_map[torch.argmax(preds).item()]}"
                        )
                        axs[i // 8, i % 8].axis("off")

                    # Show images and reset batch list
                    plt.show()
                    batch_list = []


if __name__ == "__main__":
    # The imgs_path and model_path must be changed if users would like to use it on their own machine (figure out the path to the model and images)
    model_path = f"{os.path.join(os.getcwd(), 'intel_model.pt')}"
    imgs_path = "C:\Microsoft VS Code\ADEIJ_datasets\seg_pred\seg_pred"
    show = False

    inference(model_path=model_path, imgs_path=imgs_path, show=show)
