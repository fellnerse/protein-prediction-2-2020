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
        hidden_dim=1024,
        output_dim=11,
        dropout=0.25,
        kernel_size=9,
        conv_dropout: float = 0.25,
    ):
        super(StandardAttention, self).__init__()
        self.embeddings_dim = embeddings_dim
        self.dropout_pct = dropout

        self.feature_convolution = nn.Conv1d(
            embeddings_dim, hidden_dim, kernel_size, stride=1, padding=kernel_size // 2
        )
        self.attention_convolution = nn.Conv1d(
            embeddings_dim, hidden_dim, kernel_size, stride=1, padding=kernel_size // 2
        )

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * hidden_dim, 32),
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


# todo rename run standard_attention* to smaller attention (this run does not attend to each value in input, but each feature vector)
# todo add actual self attention (with key query value)
# todo add "two way" attention (1. attend to each feature vector; 2. attend to each feature dim)


class AttentionModule(nn.Module):
    def __init__(
        self,
        embeddings_dim=1024,
        hidden_dim=1024,
        dropout=0.25,
        kernel_size=9,
        conv_dropout: float = 0.25,
    ):
        super().__init__()
        self.feature_convolution = nn.Conv1d(
            embeddings_dim, hidden_dim, kernel_size, stride=1, padding=kernel_size // 2
        )
        self.attention_convolution = nn.Conv1d(
            embeddings_dim, hidden_dim, kernel_size, stride=1, padding=kernel_size // 2
        )

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            # nn.BatchNorm1d(32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, 1, sequence_length]

        o1 = torch.sum(
            o * self.softmax(attention), dim=-1
        )  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return o  # [batchsize, output_dim]


class StackedAttention(PLModel):
    def __init__(
        self,
        embeddings_dim=1024,
        hidden_dim=1024,
        output_dim=11,
        dropout=0.25,
        kernel_size=9,
        conv_dropout: float = 0.25,
    ):
        super().__init__()
        self.at0 = AttentionModule(
            embeddings_dim=embeddings_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
            conv_dropout=conv_dropout,
        )
        self.at1 = AttentionModule(
            embeddings_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
            conv_dropout=conv_dropout,
        )

        self.pooling = nn.AdaptiveAvgPool2d(hidden_dim)

        self.output = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        at0 = self.at0(x)
        at1 = self.at1(self.pooling(x.permute(0, 2, 1)))
        return self.output(torch.cat([at0, at1], dim=-1))


# todo use fixed max, min and avg -> do attention on those maybe?
