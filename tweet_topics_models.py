import torch
import transformers


class BoW(torch.nn.Module):
    def __init__(self, vocab_size, hidden_layers, activation, num_classes=6):
        super().__init__()

        # Build network using the Sequential class
        self.layers = [torch.nn.Linear(vocab_size, hidden_layers[0]), activation]

        # Loop through the hidden layers and add them to the network, and then add the activation function in between
        for i in range(len(hidden_layers) - 1):
            self.layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.layers.append(activation)
            self.layers.append(torch.nn.Dropout(p=0.3))

        self.layers.append(torch.nn.Linear(hidden_layers[-1], num_classes))

        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


# Bidirectional LSTM implementation
class LSTM(torch.nn.Module):
    def __init__(
        self, vocab_size, n_embed, pad_index, hidden_size, num_classes=6
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=n_embed, padding_idx=pad_index
        )

        self.lstm = torch.nn.LSTM(
            input_size=n_embed,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
        )

        self.out = torch.nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # Embed each token (B, T, C)
        x, _ = self.lstm(x)  # Apply LSTM (B, T, C)
        x = torch.amax(x, dim=1)  # Reduce sequence dim (B, C)
        x = self.out(x)  # (B, 2)

        return x
