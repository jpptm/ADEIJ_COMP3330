from torch.utils.data import DataLoader
from intel_dataloader import IntelDataLoader, IntelTestLoader
import export
import pdb # for debugging
import torch
from models.cv_model import CVModel


def test(csv_path, model, device):

	test_data = IntelTestLoader(csv_path)
	test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

	truth = []
	preds = []

	# Keep track of validation loss
	val_loss = 0
	correct = 0
	total = 0

	epoch = 0

	# confusion_mat = confusion_matrix(truth, preds)
	# acc = accuracy_score(truth, preds)

	# precision_global = precision_score(truth, preds, average="micro")
	# precision_mean = precision_score(truth, preds, average="macro")

	# recall_global = recall_score(truth, preds, average="micro")
	# recall_mean = recall_score(truth, preds, average="macro")

	# avg_loss = val_loss / len(val_loader)
	history=None
	export.Export(model, device, history, test_loader)

def main(data_path, hidden_size, name, kind):
	# Set device - GPU if available, else CPU
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[INFO]: USING {str(device).upper()} DEVICE")

	# Create model and optimiser
	model = CVModel(num_classes=6, hidden_size=hidden_size, kind=kind, name=name).to(device)
	# Load the state from the model path
	model.load_state_dict(torch.load(data_path["model_path"], map_location=device))

	test(data_path["test_csv"], model, device)


if __name__ == "__main__":
	# Set random seed
	torch.manual_seed(0)

	# Define hyperparameters
	data_paths = {
		"test_csv": "./../ADEIJ_datasets/seg_pred_labels.csv",
		"model_path": "./outputs/resnet50_150/resnet50_150_model.pt"
	}

	# model settings
	# kinds supported: resnet50, resnet18, vgg, efficientnet
	# make sure kind and hidden_size match the model you want to use
	kind = 'resnet50'
	hidden_size = 150
	training = False

    # define model path if not training or training from scratch
	input_map = {
		"data_path": data_paths,
		"hidden_size": hidden_size,
		"name": f"{kind}_{hidden_size}",
		"kind": kind
	}

	# Run main function
	main(**input_map)