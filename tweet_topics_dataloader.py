import os
import pickle

import datasets
import torch
import torchtext
from tqdm import tqdm


class TweetTopicsDataLoader:

    """
    Data loader for q2 - tweet topics

    Returns:
    TweetTopicsDataLoader instance

    """

    def __init__(self, splits=["train_coling2022", "test_coling2022"], batch_size=32):
        self.splits = splits
        dataset = "cardiffnlp/tweet_topic_single"

        self.train_all = datasets.load_dataset(dataset, split=self.splits[0])
        self.test = datasets.load_dataset(dataset, split=self.splits[1])

        # Split dataset, 80% train, 20% test - 20% of the 80% is valdation data
        self.train_val = self.train_all.train_test_split(test_size=0.2, shuffle=True)

        self.train = self.train_val["train"]
        self.val = self.train_val["test"]

        # print(self.train.features)
        # print(self.train[0])

        # Find max length of the tweets
        # lens = []
        # for tweet in tqdm(self.train_all, desc="Finding max tweet length"):
        #     print(tweet)
        #     input()
        #     current_tweet = tweet["text"].split(" ")
        #     print(current_tweet)
        #     lens.append(len(current_tweet))
        # print(max(lens))
        # Max length is 69 apparently

        # Tokenise all data
        self.tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
        self.train = self.train.map(
            function=self.tokenize_data,
            fn_kwargs={"tokenizer": self.tokenizer, "max_length": 70},
        )
        self.val = self.val.map(
            function=self.tokenize_data,
            fn_kwargs={"tokenizer": self.tokenizer, "max_length": 70},
        )
        self.test = self.test.map(
            function=self.tokenize_data,
            fn_kwargs={"tokenizer": self.tokenizer, "max_length": 70},
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

        # Format data
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

        count = 0
        for batch in self.val:
            print(batch["ids"])
            print(batch["label"])
            if count == 2:
                break
            count += 1

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


if __name__ == "__main__":
    # Debugging
    TweetTopicsDataLoader()

# TODO add extra logic for bag of words and bag of n grams
# TODO account for transformers too, or make an extra class if it gets too convoluted and yuck
