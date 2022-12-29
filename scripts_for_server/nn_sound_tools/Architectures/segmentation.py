import torch
import torch.nn as nn

class baseline_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=101,
                      stride=20, padding=50),
            nn.Dropout(p=0.7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(num_features=64),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=101,
                      stride=20, padding=50),
            nn.Dropout(p=0.7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(num_features=128),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=101,
                      stride=20, padding=50),
            nn.Dropout(p=0.7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(num_features=256),
        )

        self.convT_layers = nn.Sequential(
            nn.ConvTranspose1d(in_channels=256,
                               out_channels=128,
                               kernel_size=101,
                               stride=20,
                               padding=50,
                               output_padding=19),
            nn.Dropout(p=0.7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(num_features=128),
            nn.ConvTranspose1d(in_channels=128,
                               out_channels=64,
                               kernel_size=101,
                               stride=20,
                               padding=50,
                               output_padding=19),
            nn.Dropout(p=0.7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(num_features=64),
            nn.ConvTranspose1d(in_channels=64,
                               out_channels=1,
                               kernel_size=101,
                               stride=20,
                               padding=50,
                               output_padding=19),
        )

    def forward(self, x):
        return self.convT_layers(self.conv_layers(x))

class mfcc_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=(20, 5),
                      stride=(10, 2),
                      padding=(10, 2)),
            nn.Dropout(p=0.7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1)),
            nn.Dropout(p=0.7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(13, 3),
                      stride=(1, 2),
                      padding=(0, 1)),
            nn.Dropout(p=0.7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(num_features=256),
        )

        self.convT_layers = nn.Sequential(
            nn.ConvTranspose1d(in_channels=256,
                               out_channels=128,
                               kernel_size=101,
                               stride=20,
                               padding=50,
                               output_padding=19),
            nn.Dropout(p=0.7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(num_features=128),
            nn.ConvTranspose1d(in_channels=128,
                               out_channels=64,
                               kernel_size=101,
                               stride=20,
                               padding=50,
                               output_padding=19),
            nn.Dropout(p=0.7),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(num_features=64),
            nn.ConvTranspose1d(in_channels=64,
                               out_channels=1,
                               kernel_size=101,
                               stride=10,
                               padding=50,
                               output_padding=9),
        )

    def forward(self, x):
        return self.convT_layers(torch.squeeze(self.conv_layers(x), -2))

class eGMAPSv02_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.7)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        # convolution
        self.conv1 = nn.Conv2d(in_channels=1,
                                   out_channels=64,
                                   kernel_size=(5, 20),
                                   stride=(2, 6),
                                   padding=(2, 10))
        self.conv1_batch_norm = nn.BatchNorm2d(num_features=64)

        self.conv2 = nn.Conv2d(in_channels=64,
                                   out_channels=128,
                                   kernel_size=(3, 3),
                                   stride=(2, 2),
                                   padding=(1, 1))
        self.conv2_batch_norm = nn.BatchNorm2d(num_features=128)

        self.conv3 = nn.Conv2d(in_channels=128,
                                   out_channels=256,
                                   kernel_size=(6, 13),
                                   stride=(2, 2),
                                   padding=(0, 1))
        self.conv3_batch_norm = nn.BatchNorm2d(num_features=256)

        # deconvolution
        self.deconv1 = nn.ConvTranspose1d(in_channels=256,
                                              out_channels=128,
                                              kernel_size=101,
                                              stride=10,
                                              padding=50,
                                              output_padding=9)
        self.deconv1_batch_norm = nn.BatchNorm1d(num_features=128)

        self.deconv2 = nn.ConvTranspose1d(in_channels=128,
                                              out_channels=64,
                                              kernel_size=101,
                                              stride=12,
                                              padding=50,
                                              output_padding=11)
        self.deconv2_batch_norm = nn.BatchNorm1d(num_features=64)

        self.deconv3 = nn.ConvTranspose1d(in_channels=64,
                                              out_channels=1,
                                              kernel_size=101,
                                              stride=20,
                                              padding=50,
                                              output_padding=19)

    def forward(self, x):
            # Convolution
        x = self.conv1_batch_norm(self.activation(self.dropout(self.conv1(x))))
        x = self.conv2_batch_norm(self.activation(self.dropout(self.conv2(x))))
        x = self.conv3_batch_norm(self.activation(self.dropout(self.conv3(x))))

        # Deconvolution
        x = torch.squeeze(x, -2)
        x = self.deconv1_batch_norm(self.activation(self.dropout(self.deconv1(x))))
        x = self.deconv2_batch_norm(self.activation(self.dropout(self.deconv2(x))))

        return self.deconv3(x)
