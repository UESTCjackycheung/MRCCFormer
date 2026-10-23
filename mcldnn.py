import torch
import torch.nn as nn
import torch.nn.functional as F

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
