import datasets
import numpy as np
import torch
import torchtext
from niacin.text import en
from niacin.augment import randaugment


class TweetTopicsDataLoader:

    """
    Data loader for q2 - tweet topics
    Can account for BoW, LSTMs and transformers

    Returns:
    TweetTopicsDataLoader instance

    """

    def __init__(
        self,
        splits=["train_coling2022", "test_coling2022"],
        batch_size=32,
        BoW=False,
        augment=False,
    ):
        self.splits = splits
        dataset = "cardiffnlp/tweet_topic_single"

        self.train_all = datasets.load_dataset(dataset, split=self.splits[0])
        self.test = datasets.load_dataset(dataset, split=self.splits[1])

        if augment:
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

            augmented_train_data = {
                "text": [],
                "label": [],
                "date": [],
                "id": [],
                "label_name": [],
            }
            new_text = ""
            for i, (text, label, date, id, label_name) in enumerate(
                zip(
                    self.train_all["text"],
                    self.train_all["label"],
                    self.train_all["date"],
                    self.train_all["id"],
                    self.train_all["label_name"],
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

            augmented_dataset = augmented_dataset.cast(self.train_all.features)
            full_train = datasets.concatenate_datasets(
                [self.train_all, augmented_dataset]
            )

            self.train_val = full_train

        # Split dataset, 80% train, 20% test - 20% of the 80% is valdation data
        self.train_val = self.train_all.train_test_split(test_size=0.2, shuffle=True)

        self.train = self.train_val["train"]
        self.val = self.train_val["test"]

        # Tokenise all data - set max length to 300 words. A tweet's max length is 280 characters so 500 words should be safe
        self.tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
        self.train = self.train.map(
            function=self.tokenize_data,
            fn_kwargs={"tokenizer": self.tokenizer, "max_length": 500},
        )
        self.val = self.val.map(
            function=self.tokenize_data,
            fn_kwargs={"tokenizer": self.tokenizer, "max_length": 500},
        )
        self.test = self.test.map(
            function=self.tokenize_data,
            fn_kwargs={"tokenizer": self.tokenizer, "max_length": 500},
        )

        self.vocab = torchtext.vocab.build_vocab_from_iterator(
            self.train["tokens"], min_freq=5, specials=["<unk>", "<pad>"]
        )
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.pad_index = self.vocab["<pad>"]

        # Numericalise data
        self.train = self.train.map(
            self.numericalize_data, fn_kwargs={"vocab": self.vocab}
        )
        self.val = self.val.map(self.numericalize_data, fn_kwargs={"vocab": self.vocab})
        self.test = self.test.map(
            self.numericalize_data, fn_kwargs={"vocab": self.vocab}
        )

        if BoW:
            # Multi hot encode data
            self.train = self.train.map(
                self.multi_hot, fn_kwargs={"num_classes": len(self.vocab)}
            )
            self.val = self.val.map(
                self.multi_hot, fn_kwargs={"num_classes": len(self.vocab)}
            )
            self.test = self.test.map(
                self.multi_hot, fn_kwargs={"num_classes": len(self.vocab)}
            )

            # Format data
            self.train = self.train.with_format(
                type="torch", columns=["multi_hot", "label"]
            )
            self.val = self.val.with_format(
                type="torch", columns=["multi_hot", "label"]
            )
            self.test = self.test.with_format(
                type="torch", columns=["multi_hot", "label"]
            )

            # Chuck BoW ready data to DataLoader
            self.train = torch.utils.data.DataLoader(
                self.train, batch_size=batch_size, shuffle=True
            )
            self.val = torch.utils.data.DataLoader(self.val, batch_size=batch_size)
            self.test = torch.utils.data.DataLoader(self.test, batch_size=batch_size)

        else:
            self.train = self.train.with_format(type="torch", columns=["ids", "label"])
            self.val = self.val.with_format(type="torch", columns=["ids", "label"])
            self.test = self.test.with_format(type="torch", columns=["ids", "label"])

            # Collate data
            self.train = torch.utils.data.DataLoader(
                self.train,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=self.collate,
            )
            self.val = torch.utils.data.DataLoader(
                self.val,
                batch_size=batch_size,
                collate_fn=self.collate,
            )
            self.test = torch.utils.data.DataLoader(
                self.test,
                batch_size=batch_size,
                collate_fn=self.collate,
            )

    def tokenize_data(self, example, tokenizer, max_length):
        tokens = tokenizer(example["text"][:max_length])
        return {"tokens": tokens}

    def numericalize_data(self, example, vocab):
        ids = [vocab[token] for token in example["tokens"]]
        return {"ids": ids}

    # For LSTMS - if Bag of words or Bag of Ngrams then we need some extra logic
    def collate(self, batch):
        batch_ids = [b["ids"] for b in batch]
        batch_ids = torch.nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=self.pad_index, batch_first=True
        )
        batch_label = [b["label"] for b in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "label": batch_label}

        return batch

    def multi_hot(self, example, num_classes):
        encoded = np.zeros((num_classes,))
        encoded[example["ids"]] = 1
        return {"multi_hot": encoded}


# if __name__ == "__main__":
#     # Debugging
#     TweetTopicsDataLoader(BoW=True)
