import os
import pickle

import datasets
import torch


class TweetTopicsDataLoader(torch.utils.data.Dataset):

    """
    Data loader for q2 - tweet topics

    Parameters:
    split(str): Either train_coling2022 or test_coling2022
    pickle_data(bool): If true, pickle the data so we have a local copy

    Returns:
    TweetTopicsDataLoader instance

    NOTE: If pickle_data is set to True, the data will be pickled in the current working directory for a local copy
    NOTE: the "text" field contains the raw data, the "label_name" contains the topic and the "label" contains the topic index
    """

    def __init__(self, split, pickle_data=False):
        self.split = split

        self.data = datasets.load_dataset(
            "cardiffnlp/tweet_topic_single", split=self.split
        )

        # If pickle data is true, pickle the data so we have a local copy
        if pickle_data:
            data_list = [elem for elem in self.data]
            with open(f"{os.path.join(os.getcwd(), self.split)}.pickle", "wb") as f:
                pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass


# TODO: Encoding the data to vectors that can be used by the models
# TODO: Tokenizers, transformers, lstms, etc.

# print(TweetTopicsDataLoader(split="train_coling2022", pickle_data=False).__len__())
