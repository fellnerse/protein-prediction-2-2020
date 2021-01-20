import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from protein_prediction_2_2020.neural_networks.models import PLModel


class LightAttention(PLModel):
    def __init__(
            self,
            embeddings_dim=1024,
            output_dim=11,
            dropout=0.25,
            kernel_size=9,
            conv_dropout: float = 0.25,
    ):
        super(LightAttention, self).__init__()
        self.embeddings_dim = embeddings_dim
        self.dropout_pct = dropout

        self.feature_convolution = nn.Conv1d(
            embeddings_dim,
            embeddings_dim,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.attention_convolution = nn.Conv1d(
            embeddings_dim,
            embeddings_dim,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            # nn.BatchNorm1d(32),
        )

        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.
        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.attention_convolution(
            x
        )  # [batch_size, embeddings_dim, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        # attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        o1 = torch.sum(
            o * self.softmax(attention), dim=-1
        )  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, output_dim]


class LightAttentionOnlyMax(PLModel):
    def __init__(
            self,
            embeddings_dim=1024,
            output_dim=11,
            dropout=0.25,
            kernel_size=9,
            conv_dropout: float = 0.25,
    ):
        super(LightAttentionOnlyMax, self).__init__()
        self.embeddings_dim = embeddings_dim
        self.dropout_pct = dropout

        self.feature_convolution = nn.Conv1d(
            embeddings_dim,
            embeddings_dim,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            # nn.BatchNorm1d(32),
        )

        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.
        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)  # [batch_size, embeddings_dim, sequence_length]ize, embeddings_dim, sequence_length]

        o, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, output_dim]


class BetterAttention(PLModel):
    def __init__(self, embeddings_dim=1024, kernel_size=32):
        super(BetterAttention, self).__init__()
        self.attention_convolution = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size, stride=1, padding=kernel_size // 2),  # -> 1024
            nn.MaxPool1d(2),  # -> 512
            nn.LeakyReLU(),
            nn.Conv1d(1, 1, kernel_size, stride=1, padding=kernel_size // 2),  # -> 512
            nn.MaxPool1d(2),  # -> 256
            nn.LeakyReLU(),
            nn.Conv1d(1, 1, kernel_size, stride=1, padding=kernel_size // 2),  # -> 256
            nn.MaxPool1d(2),  # -> 128
            nn.LeakyReLU(),
            nn.Conv1d(1, 1, kernel_size, stride=1, padding=kernel_size // 2),  # -> 128
            nn.MaxPool1d(2),  # -> 64
            nn.LeakyReLU(),
            nn.Conv1d(
                1, 1, kernel_size // 2, stride=1, padding=kernel_size // 4
            ),  # -> 64
            nn.MaxPool1d(2),  # -> 32
            nn.LeakyReLU(),
            nn.Conv1d(
                1, 1, kernel_size // 4, stride=1, padding=kernel_size // 8
            ),  # -> 32
            nn.MaxPool1d(2),  # -> 16
            nn.LeakyReLU(),
            nn.Conv1d(
                1, 1, kernel_size // 8, stride=1, padding=kernel_size // 16
            ),  # -> 8
            nn.MaxPool1d(2),  # -> 4
            nn.LeakyReLU(),
            nn.Conv1d(
                1, 1, kernel_size // 16, stride=1, padding=kernel_size // 32
            ),  # -> 4
            nn.MaxPool1d(2),  # -> 2
            nn.LeakyReLU(),
            nn.Conv1d(1, 1, kernel_size // 32, stride=1, padding=0),  # -> 3
            nn.LeakyReLU(),
            nn.Conv1d(1, 1, 4, stride=1, padding=0),  # -> 1
        )
        self.feature_convolution = nn.Identity()

        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 32),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(32),
        )

        self.output = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.attention_convolution(
            x.permute((2, 0, 1))
        )  # [sequence_length, 1, 1]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        # attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        o1 = torch.sum(
            o * F.softmax(attention.permute(2, 1, 0)), dim=-1
        )  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, output_dim]


class LightAttentionNoMax(LightAttention):
    def __init__(self, **kwargs):
        super(LightAttentionNoMax, self).__init__(**kwargs)
        self.linear = nn.Sequential(
            nn.Linear(self.embeddings_dim, 32), nn.Dropout(self.dropout_pct), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.attention_convolution(
            x
        )  # [batch_size, embeddings_dim, sequence_length]
        o = torch.sum(
            o * self.softmax(attention), dim=-1
        )  # [batchsize, embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, output_dim]


class StandardAttention(PLModel):
    def __init__(
            self,
            embeddings_dim=1024,
            output_dim=11,
            dropout=0.25,
            kernel_size=9,
            conv_dropout: float = 0.25,
    ):
        super(StandardAttention, self).__init__()
        self.embeddings_dim = embeddings_dim
        self.dropout_pct = dropout

        self.feature_convolution = nn.Conv1d(
            embeddings_dim,
            embeddings_dim,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.attention_convolution = nn.Conv1d(
            embeddings_dim, 1, kernel_size, stride=1, padding=kernel_size // 2
        )

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            # nn.BatchNorm1d(32),
        )

        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.
        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, 1, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        # attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        o1 = torch.sum(
            o * self.softmax(attention), dim=-1
        )  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, output_dim]


class StandardAttentionWOFeatureConv(StandardAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_convolution = nn.Identity()
        # this leads to o2 being the the max of input


class SelfAttention(PLModel):
    def __init__(self,
                 input_dim=1024,
                 embeddings_dim=256,
                 attention_hidden_dim=256,
                 hidden_dim=256,
                 output_dim=11,
                 dropout=0.2,
                 kernel_size=9,
                 attn_dropout: float = 0.1):
        super(SelfAttention, self).__init__()
        self.embeddings_dim = embeddings_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.dropout_pct = dropout

        self.feature_convolution = nn.Conv1d(input_dim, embeddings_dim, kernel_size, stride=1, padding=kernel_size // 2)

        self.attention_weights = nn.Linear(embeddings_dim, 2 * attention_hidden_dim + embeddings_dim)
        self.attention_layer_norm = nn.LayerNorm(embeddings_dim)

        self.feed_forward = nn.Sequential(OrderedDict({
            'linear_1': nn.Linear(embeddings_dim, hidden_dim),
            'relu': nn.ReLU(),
            'linear_2': nn.Linear(hidden_dim, embeddings_dim),
        }))
        self.feed_forward_norm = nn.LayerNorm(embeddings_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.linear = nn.Sequential(
            nn.Linear(embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            # nn.BatchNorm1d(32),
        )

        self.output = nn.Linear(32, output_dim)

    def positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze()
        pe = torch.zeros_like(x).cuda()
        position = torch.arange(0, x.shape[0], dtype=torch.float).unsqueeze(1).cuda()
        div_term = torch.exp(torch.arange(0, x.shape[1], 2).float() * (-math.log(10000.0) / x.shape[1]))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def attention(self, x) -> torch.Tensor:
        v = self.attention_weights(x)  # [sequence_length, 2*attention_hidden_dim + embeddings_dim]
        k, q, v = torch.split(v, [self.attention_hidden_dim, self.attention_hidden_dim, self.embeddings_dim], dim=1)
        # [sequence_length, attention_hidden_dim], [sequence_length, attention_hidden_dim], [sequence_length, embeddings_dim]
        k = q @ k.t()  # [sequence_length, sequence_length]
        k = k / math.sqrt(self.attention_hidden_dim)  # [sequence_length, sequence_length]
        v = F.softmax(k) @ v  # [sequence_length, embeddings_dim]
        x = self.attention_layer_norm(x + self.attn_dropout(v))  # [sequence_length, embeddings_dim]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_convolution(x)  # [batch_size, embedding_dim, sequence_length]
        x = x.squeeze()  # [embedding_dim, sequence_length]
        x = x.t()  # [sequence_length, embedding_dim]
        x = self.attention(x)  # [sequence_length, embedding_dim]
        x = self.feed_forward_norm(x + self.attn_dropout(self.feed_forward(x)))  # [sequence_length, embedding_dim]
        x = x.max(dim=0)[0]
        x = self.linear(x)
        x = self.output(x)
        return x
# todo rename run standard_attention* to smaller attention (this run does not attend to each value in input, but each feature vector)
# todo add "two way" attention (1. attend to each feature vector; 2. attend to each feature dim)
