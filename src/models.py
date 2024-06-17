import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            TimeConvBlock(in_channels, hid_dim),
            TimeConvBlock(hid_dim, hid_dim),
            TimeConvBlock(hid_dim, hid_dim),
            TimeConvBlock(hid_dim, hid_dim),
            TimeConvBlock(hid_dim, hid_dim),
            FreqConvBlock(hid_dim, hid_dim),
            FreqConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class TimeConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv2d(in_dim, out_dim, (1, 5), (1, 2), (0, 2))
        self.conv1 = nn.Conv2d(out_dim, out_dim, (1, 3), (1, 1), (0, 1))

        self.conv_short = nn.Conv2d(in_dim, out_dim, (1, 3), (1, 2), (0, 1))

        self.batchnorm0 = nn.BatchNorm2d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = F.gelu(self.batchnorm0(self.conv0(X)))
        Y = F.gelu(self.batchnorm1(self.conv1(Y)))
        return self.dropout(Y + self.conv_short(X))


class FreqConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv2d(in_dim, out_dim, (3, 1), (1, 1), (1, 0))
        self.conv1 = nn.Conv2d(out_dim, out_dim, (3, 1), (1, 1), (1, 0))

        self.conv_short = nn.Conv2d(in_dim, out_dim, (1, 1))

        self.batchnorm0 = nn.BatchNorm2d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = F.gelu(self.batchnorm0(self.conv0(X)))
        Y = F.gelu(self.batchnorm1(self.conv1(Y)))
        return self.dropout(Y + self.conv_short(X))


class Wavelet(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer("w1", torch.tensor([0.5, 0.5]).reshape(1, 1, 1, 2))
        self.register_buffer("w2", torch.tensor([0.5, -0.5]).reshape(1, 1, 1, 2))

    def forward(self, x):
        B, C, _, T = x.shape
        y1 = torch.nn.functional.conv2d(x.reshape(B*C, 1, -1, T), self.w1, stride=(1, 2)).reshape(B, C, -1, T//2)
        y2 = torch.nn.functional.conv2d(x.reshape(B*C, 1, -1, T), self.w2, stride=(1, 2)).reshape(B, C, -1, T//2)

        if y1.shape[3] > 1:
            y1 = self.forward(y1).unsqueeze(4).repeat(1, 1, 1, 1, 2).reshape(y2.shape[0], y2.shape[1], -1, y2.shape[3])

        return torch.cat([y1, y2], 2)
