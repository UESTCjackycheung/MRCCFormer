import torch
import torch.nn as nn
import torch.nn.functional as F
import time

"\n《A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition》\n<https://ieeexplore.ieee.org/abstract/document/9106397>\n"


class MCLDNN(nn.Module):
    def __init__(self, num_classes, sig_size):
        super(MCLDNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=50, kernel_size=(2, 8), padding="same"
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(in_channels=1, out_channels=50, kernel_size=8),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0),
            nn.Conv1d(in_channels=1, out_channels=50, kernel_size=8),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=50, out_channels=50, kernel_size=(1, 8), padding="same"
            ),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=100, kernel_size=(2, 5)), nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=100, hidden_size=128, num_layers=2, batch_first=True
        )
        self.fc1 = nn.Sequential(nn.Linear(128, 128), nn.SELU(), nn.Dropout())
        self.fc2 = nn.Sequential(nn.Linear(128, 128), nn.SELU(), nn.Dropout())
        self.softmax = nn.Sequential(nn.Linear(128, num_classes))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        param.data.zero_()

    def forward(self, batch_x):
        conv2_in, conv3_in = (batch_x[:, 0:1], batch_x[:, 1:2])
        conv2_out, conv3_out = (self.conv2(conv2_in), self.conv3(conv3_in))
        conv2_out, conv3_out = (
            torch.unsqueeze(conv2_out, dim=2),
            torch.unsqueeze(conv3_out, dim=2),
        )
        concatenate1 = torch.concatenate((conv2_out, conv3_out), dim=2)
        conv4_out = self.conv4(concatenate1)
        batch_x = torch.unsqueeze(batch_x, dim=1)
        conv1_out = self.conv1(batch_x)
        concatenate2 = torch.concatenate((conv4_out, conv1_out), dim=1)
        conv5_out = self.conv5(concatenate2)
        conv5_out = conv5_out.permute(0, 3, 2, 1).flatten(2)
        outputs, _ = self.lstm(conv5_out)
        outputs = self.fc1(outputs[:, -1])
        outputs = self.fc2(outputs)
        outputs = self.softmax(outputs)
        return outputs


# Architecture used for 1024-sample HisarMod inputs.


class _MCLDNN1024(nn.Module):
    """`MCLDNN <https://ieeexplore.ieee.org/abstract/document/9106397>`_ backbone
    The input for LSTM is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, frame_length=128, num_classes=11):
        super(_MCLDNN1024, self).__init__()
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(2, 8), padding="same"), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0), nn.Conv1d(1, 50, kernel_size=8), nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0), nn.Conv1d(1, 50, kernel_size=8), nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(1, 8), padding="same"), nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=(2, 5), padding="valid"), nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=100, hidden_size=128, batch_first=True, num_layers=2
        )
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.Linear(128, 128),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )

    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x2 = self.conv2(x[:, :, 0, :])
        x3 = self.conv3(x[:, :, 1, :])
        x4 = self.conv4(torch.stack([x2, x3], dim=2))
        x5 = self.conv5(torch.cat([x1, x4], dim=1))
        x = torch.reshape(x5, [-1, self.frame_length - 4, 100])
        x, _ = self.lstm(x)
        x = self.classifier(x[:, -1, :])
        return x


def build_mcldnn(num_classes, signal_length):
    if signal_length == 128:
        return MCLDNN(num_classes, signal_length)
    if signal_length == 1024:
        return _MCLDNN1024(frame_length=signal_length, num_classes=num_classes)
    raise ValueError("MCLDNN supports input lengths 128 and 1024")
