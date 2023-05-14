import datasets
import numpy as np
import torch
import transformers

from tqdm import tqdm
from transformers import BertTokenizer

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

import matplotlib.pyplot as plt

# Script mostly breaks pretrained models so be careful with this one
class TransformersDataLoader:
    def __init__(
        self,
        splits=["train_coling2022", "test_coling2022"],
        batch_size=8,
        augment=False,
    ):
        self.splits = splits
        dataset = "cardiffnlp/tweet_topic_single"

        self.train_all = datasets.load_dataset(dataset, split=self.splits[0])
        self.test = datasets.load_dataset(dataset, split=self.splits[1])

        # Split dataset, 80% train, 20% test - 20% of the 80% is valdation data
        self.train_val = self.train_all.train_test_split(test_size=0.2, shuffle=True)

        self.train = self.train_val["train"]
        self.val = self.train_val["test"]

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.train_tokenised = self.train.map(
            lambda examples: self.tokenizer(
                examples["text"], padding="max_length", truncation=True, max_length=500
            )
        )

        self.val_tokenised = self.val.map(
            lambda examples: self.tokenizer(
                examples["text"], padding="max_length", truncation=True, max_length=500
            )
        )

        self.test_tokenised = self.test.map(
            lambda examples: self.tokenizer(
                examples["text"], padding="max_length", truncation=True, max_length=500
            )
        )

        self.train_tokenised = self.train_tokenised.with_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        self.val_tokenised = self.val_tokenised.with_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        self.test_tokenised = self.test_tokenised.with_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_tokenised,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate,
        )

        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_tokenised,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate,
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_tokenised,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate,
        )

    def collate(self, batch):
        batch_ids = [b["input_ids"] for b in batch]
        batch_ids = torch.stack(batch_ids)

        batch_attention = [b["attention_mask"] for b in batch]
        batch_attention = torch.stack(batch_attention)

        batch_label = [b["label"] for b in batch]
        batch_label = torch.stack(batch_label)

        batch = {
            "input_ids": batch_ids,
            "label": batch_label,
            "attention_mask": batch_attention,
        }

        return batch


class Transformer(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Load the pre-trained BERT model
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased")

        # Add a new classification layer on top of BERT's output
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # Pass the input through BERT and get the pooled output
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        # Pass the pooled output through the classifier layer
        logits = self.classifier(pooled_output)

        return logits


class Trainer:
    def __init__(self, master_data, model, optimiser, loss, num_epochs, learning_rate):
        self.train_dataloader = master_data.train_dataloader
        self.val_dataloader = master_data.val_dataloader
        self.test_dataloader = master_data.test_dataloader

        self.num_epochs = num_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimiser = optimiser(self.model.parameters(), lr=learning_rate)
        self.loss_func = loss.to(self.device)

    def run(self):
        # Logging
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1} of {self.num_epochs}")
            # Train
            train_loss, train_accuracy = self.train()
            # Evaluate
            valid_loss, valid_accuracy = self.val()
            # Log
            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            val_losses.append(valid_loss)
            val_accs.append(valid_accuracy)
            print(
                "Epoch {}: train_loss={:.4f}, train_accuracy={:.4f}, valid_loss={:.4f}, valid_accuracy={:.4f}".format(
                    epoch + 1, train_loss, train_accuracy, valid_loss, valid_accuracy
                )
            )

        # Test
        self.test()

        # Show loss and accuracy history
        plt.figure()
        plt.plot(train_losses, label="Training loss")
        plt.plot(val_losses, label="Validation loss")
        plt.legend()

        plt.figure()
        plt.plot(train_accs, label="Training accuracy")
        plt.plot(val_accs, label="Validation accuracy")
        plt.legend()
        plt.show()

    def train(self):
        self.model.train()
        losses, accuracies = [], []

        for batch in tqdm(
            self.train_dataloader,
            position=1,
            total=len(self.train_dataloader),
            leave=False,
            desc="Training",
        ):
            inputs = batch["input_ids"].to(self.device)
            attention = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            # Reset the gradients for all variables
            self.optimiser.zero_grad()

            # Forward pass
            preds = self.model(inputs, attention)
            # Calculate loss
            loss = self.loss_func(preds, labels)

            # Backward pass
            loss.backward()
            # Adjust weights
            self.optimiser.step()

            # Log
            losses.append(loss.detach().cpu().numpy())
            accuracy = (
                torch.sum(torch.argmax(preds, dim=-1) == labels) / labels.shape[0]
            )
            accuracies.append(accuracy.detach().cpu().numpy())

        return np.mean(losses), np.mean(accuracies)

    def val(self):
        self.model.eval()

        losses, accuracies = [], []
        with torch.no_grad():
            for batch in tqdm(
                self.val_dataloader,
                position=1,
                total=len(self.val_dataloader),
                leave=False,
                desc="Validating",
            ):
                inputs = batch["input_ids"].to(self.device)
                attention = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward pass
                preds = self.model(inputs, attention)
                # Calculate loss
                loss = self.loss_func(preds, labels)
                # Log
                losses.append(loss.detach().cpu().numpy())
                accuracy = (
                    torch.sum(torch.argmax(preds, dim=-1) == labels) / labels.shape[0]
                )
                accuracies.append(accuracy.detach().cpu().numpy())

        return np.mean(losses), np.mean(accuracies)

    # Test on the test dataset
    def test(self):

        truth = []
        preds = []

        val_loss = 0
        correct = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(
                self.test_dataloader,
                position=1,
                total=len(self.test_dataloader),
                leave=False,
                desc="Testing",
            ):
                # Cast tensors to device
                inputs = batch["input_ids"].to(self.device)
                attention = batch["attention_mask"].to(self.device)
                targets = batch["label"].to(self.device)

                # Calculate model output and loss
                outputs = self.model(inputs, attention)
                loss = self.loss_func(outputs, targets)

                # Keep track of loss and accuracy
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                truth.append(targets.cpu().numpy().flatten())
                preds.append(predicted.cpu().numpy().flatten())

        # Flatten outputs
        truth = [item for sublist in truth for item in sublist]
        preds = [item for sublist in preds for item in sublist]

        # Get accuracy, precision and recall
        confusion_mat = confusion_matrix(truth, preds)
        acc = accuracy_score(truth, preds)

        precision_global = precision_score(truth, preds, average="micro")
        precision_mean = precision_score(truth, preds, average="macro")

        recall_global = recall_score(truth, preds, average="micro")
        recall_mean = recall_score(truth, preds, average="macro")

        print("Confusion matrix: \n{}\n".format(confusion_mat))
        print(" Accuracy - {:.4f}".format(acc))
        print(
            "Precision - Global: {:.4f} \t Mean: {:.4f}".format(
                precision_global, precision_mean
            )
        )
        print(
            "   Recall - Global: {:.4f} \t Mean: {:.4f}".format(
                recall_global, recall_mean
            )
        )


if __name__ == "__main__":
    master_data = TransformersDataLoader()
    model = Transformer(num_classes=6)

    configs = {
        "master_data": master_data,
        "model": model,
        "optimiser": torch.optim.AdamW,
        "loss": torch.nn.CrossEntropyLoss(),
        "num_epochs": 10,
        "learning_rate": 0.0001,
    }

    trainer = Trainer(**configs)
    trainer.run()
