import torch
import torch.nn as nn
import torch.nn.functional as F
import time

"《An Efficient Deep Learning Model for Automatic Modulation Recognition Based on Parameter Estimation and Transformation》"


class PETCGDNN(nn.Module):
    def __init__(self, num_classes: int = 11, sig_size: int = 128):
        super(PETCGDNN, self).__init__()
        self.phi_fc = nn.Linear(in_features=2 * sig_size, out_features=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=75, kernel_size=(2, 8)), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=75, out_channels=25, kernel_size=(1, 5)), nn.ReLU()
        )
        self.gru = nn.GRU(input_size=25, hidden_size=128, batch_first=True)
        self.classify = nn.Sequential(nn.Linear(128, num_classes))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        param.data.zero_()

    def forward(self, batch_x: torch.Tensor):
        part1_batch_x = batch_x.flatten(start_dim=1)
        part1_batch_x = self.phi_fc(part1_batch_x)
        phi = self.linear_activation(part1_batch_x)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        batch_i = batch_x[:, 0]
        batch_q = batch_x[:, 1]
        transformation_add = batch_i * cos_phi + batch_q * sin_phi
        transformation_minus = batch_q * cos_phi - batch_i * sin_phi
        transformation_add = torch.unsqueeze(transformation_add, dim=1)
        transformation_minus = torch.unsqueeze(transformation_minus, dim=1)
        batch_transformation = torch.concatenate(
            (transformation_add, transformation_minus), dim=1
        )
        batch_transformation = torch.unsqueeze(batch_transformation, dim=1)
        part3_batch_x = self.conv1(batch_transformation)
        part3_batch_x = self.conv2(part3_batch_x)
        part3_batch_x = part3_batch_x.permute(0, 3, 2, 1).flatten(2)
        part3_batch_x, _ = self.gru(part3_batch_x)
        batch_y = self.classify(part3_batch_x[:, -1])
        return batch_y

    def linear_activation(self, x):
        y = x
        return y


# Architecture used for 1024-sample HisarMod inputs.


class PET(nn.Module):
    def __init__(self, frame_length=128):
        super(PET, self).__init__()
        self.p1 = nn.Sequential(nn.Flatten(), nn.Linear(frame_length * 2, 1))

    def forward(self, x):
        p1_x = self.p1(x)
        sin_x = torch.sin(p1_x)
        cos_x = torch.cos(p1_x)
        x11 = x[:, :, 0] * cos_x
        x12 = x[:, :, 1] * sin_x
        x21 = x[:, :, 0] * sin_x
        x22 = x[:, :, 1] * cos_x
        y1 = x11 + x12
        y2 = x21 - x22
        y1 = torch.unsqueeze(y1, 2)
        y2 = torch.unsqueeze(y2, 2)
        x2 = torch.cat([y1, y2], dim=2)
        x2 = torch.transpose(x2, 1, 2)
        x2 = torch.unsqueeze(x2, 1)
        return x2


class _PETCGDNN1024(nn.Module):
    """`PETCGDNN <https://ieeexplore.ieee.org/abstract/document/9507514>`_ backbone
    The input for PETCGDNN is an N*L*2 frame
    Args:
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(self, num_classes=26, frame_length=1024, hidden_size=128):
        super(_PETCGDNN1024, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            PET(frame_length=frame_length),
            nn.Conv2d(1, 75, kernel_size=(2, 8), padding="valid"),
            nn.ReLU(inplace=True),
            nn.Conv2d(75, 25, kernel_size=(1, 5), padding="valid"),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(input_size=25, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.transpose(x, 2, 1)
        x = self.features(x)
        x = torch.squeeze(x, dim=2)
        x = torch.transpose(x, 1, 2)
        x, _ = self.gru(x)
        x = self.classifier(x[:, -1, :])
        return x


def build_petcgdnn(num_classes, signal_length):
    if signal_length == 128:
        return PETCGDNN(num_classes, signal_length)
    if signal_length == 1024:
        return _PETCGDNN1024(num_classes, signal_length)
    raise ValueError("PET-CGDNN supports input lengths 128 and 1024")
