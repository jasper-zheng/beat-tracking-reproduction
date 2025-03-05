import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_channels=1):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=(3, 3), padding=(1, 0))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3))

        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 0))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3))

        self.conv3 = nn.Conv2d(16, 16, kernel_size=(1, 8))

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # print(x.shape)
        x = self.elu(self.conv1(x))
        x = self.pool1(x)

        x = self.elu(self.conv2(x))
        x = self.pool2(x)

        x = self.elu(self.conv3(x))
        x = self.dropout(x)

        # x = x.squeeze(2)
        x = x.view(-1, x.shape[1], x.shape[2])
        # print(x.shape)
        return x


class TCNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()

        padding = (kernel_size - 1) * dilation  # causal
        self.padding = padding
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.elu = nn.ELU()

    def forward(self, x):

        x = self.conv1(x)
        x = x[:, :, :-self.padding].contiguous()
        x = self.elu(x)
        # x = self.dropout(x)
        return x

class TCN(nn.Module):
    def __init__(self, input_dim, num_filters, kernel_size, num_layers, dropout=0.1):
        super(TCN, self).__init__()

        layers = []
        for i in range(num_layers):
            dilation = 2**i
            in_channels = input_dim if i == 0 else num_filters
            layers.append(TCNBlock(in_channels, num_filters, kernel_size, dilation))

        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Conv1d(num_filters, 1, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        In: (batch_size, input_dim, time_steps)
        Out: (batch_size, 1, time_steps)
        """
        x = self.network(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)


class BeatTrackingNet(nn.Module):
    def __init__(self, input_dim, num_filters, kernel_size, num_layers, dropout=0.1):
        super(BeatTrackingNet, self).__init__()
        self.conv_block = ConvBlock()
        self.tcn = TCN(16, num_filters, kernel_size, num_layers, dropout)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.tcn(x).squeeze(1)
        return x


