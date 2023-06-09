import os

import datasets
import evaluate
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from transformers import (
    Trainer,
    TrainingArguments,
)

from niacin.text import en
from niacin.augment import randaugment

# Split dataset
splits = ["train_coling2022", "test_coling2022"]
dataset = "cardiffnlp/tweet_topic_single"

train_all = datasets.load_dataset(dataset, split=splits[0])
test = datasets.load_dataset(dataset, split=splits[1])

# Apply data augmentation
augmentor = randaugment.RandAugment(
    [
        en.add_synonyms,
        en.add_hyponyms,
        en.add_misspelling,
        en.swap_words,
        en.add_contractions,
        # en.add_whitespace,
    ],
    n=3,
    m=10,
    shuffle=False,
)


augmented_train_data = {"text": [], "label": [], "date": [], "id": [], "label_name": []}
new_text = ""
for i, (text, label, date, id, label_name) in enumerate(
    zip(
        train_all["text"],
        train_all["label"],
        train_all["date"],
        train_all["id"],
        train_all["label_name"],
    )
):
    # print(text)
    for tx in augmentor:
        new_text = tx(text)
        # print("here")

    # Sometimes the augmentation doesn't work, so we need to check if the text has changed
    if text != new_text:
        # Quick and sloppy
        augmented_train_data["text"].append(new_text)
        augmented_train_data["label"].append(label)
        augmented_train_data["date"].append(date)
        augmented_train_data["id"].append(id)
        augmented_train_data["label_name"].append(label_name)

        # print(augmented_train_data[count]["text"], "\n")
        # count += 1

# Cast augmented data to datasets object
augmented_dataset = datasets.Dataset.from_dict(
    {
        "text": augmented_train_data["text"],
        "date": augmented_train_data["date"],
        "label": augmented_train_data["label"],
        "label_name": augmented_train_data["label_name"],
        "id": augmented_train_data["id"],
    }
)

augmented_dataset = augmented_dataset.cast(train_all.features)
full_train = datasets.concatenate_datasets([train_all, augmented_dataset])

print(full_train)

# Split concatenated dataset
train_val = full_train.train_test_split(test_size=0.2, shuffle=True)

train = train_val["train"]
val = train_val["test"]


# Initializing a model (with random weights) from the configuration
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Load the pre-trained BERT model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=6)

# for param in model.base_model.parameters():
#     param.requires_grad = False

# Tokenising function to be mapped
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


train_token = train.map(tokenize_function, batched=True)
val_token = val.map(tokenize_function, batched=True)
test_token = test.map(tokenize_function, batched=True)

training_args = TrainingArguments(output_dir=os.getcwd())
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir=os.getcwd(),
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    # load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_token,
    eval_dataset=val_token,
    compute_metrics=compute_metrics,
)

# Train and evaluatemodel
trainer.train()

# Evaluate on the test set
predictions = trainer.predict(test_token)

# Get predicted labels from model
pred_labels = np.argmax(predictions.predictions, axis=-1)

# Get true labels from test set
true_labels = test["label"]

# Compute evaluation metrics
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average="weighted")
recall = recall_score(true_labels, pred_labels, average="weighted")
conf_matrix = confusion_matrix(true_labels, pred_labels)

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Confusion matrix:\n{conf_matrix}")
