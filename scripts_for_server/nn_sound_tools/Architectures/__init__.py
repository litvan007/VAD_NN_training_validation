  1 import torch
  2 import torch.nn as nn
  3
  4 class baseline_model(nn.Module):
  5     def __init__(self):
  6         super().__init__()
  7         self.conv_layers = nn.Sequential(
  8             nn.Conv1d(in_channels=1, out_channels=64, kernel_size=101,
  9                       stride=20, padding=50),
 10             nn.Dropout(p=0.7),
 11             nn.LeakyReLU(negative_slope=0.2),
 12             nn.BatchNorm1d(num_features=64),
 13
 14             nn.Conv1d(in_channels=64, out_channels=128, kernel_size=101,
 15                       stride=20, padding=50),
 16             nn.Dropout(p=0.7),
 17             nn.LeakyReLU(negative_slope=0.2),
 18             nn.BatchNorm1d(num_features=128),
 19
 20             nn.Conv1d(in_channels=128, out_channels=256, kernel_size=101,
 21                       stride=20, padding=50),
 22             nn.Dropout(p=0.7),
 23             nn.LeakyReLU(negative_slope=0.2),
 24             nn.BatchNorm1d(num_features=256),
 25         )
 26
 27         self.convT_layers = nn.Sequential(
 28             nn.ConvTranspose1d(in_channels=256,
 29                                out_channels=128,
 30                                kernel_size=101,
 31                                stride=20,
 32                                padding=50,
 33                                output_padding=19),
 34             nn.Dropout(p=0.7),
 35             nn.LeakyReLU(negative_slope=0.2),
 36             nn.BatchNorm1d(num_features=128),
 37             nn.ConvTranspose1d(in_channels=128,
 38                                out_channels=64,
 39                                kernel_size=101,
 40                                stride=20,
 41                                padding=50,
 42                                output_padding=19),
 43             nn.Dropout(p=0.7),
 44             nn.LeakyReLU(negative_slope=0.2),
 45             nn.BatchNorm1d(num_features=64),
 46             nn.ConvTranspose1d(in_channels=64,
 47                                out_channels=1,
 48                                kernel_size=101,
 49                                stride=20,
 50                                padding=50,
 51                                output_padding=19),
 52         )
 53
 54     def forward(self, x):
 55         return self.convT_layers(self.conv_layers(x))
 56
 57 class mfcc_model(nn.Module):
 58     def __init__(self):
 59         super().__init__()
 60         self.conv_layers = nn.Sequential(
  1 import torch
  2 import torch.nn as nn
  3
  4 class baseline_model(nn.Module):
  5     def __init__(self):
  6         super().__init__()
  7         self.conv_layers = nn.Sequential(
  8             nn.Conv1d(in_channels=1, out_channels=64, kernel_size=101,
  9                       stride=20, padding=50),
 10             nn.Dropout(p=0.7),
 11             nn.LeakyReLU(negative_slope=0.2),
 12             nn.BatchNorm1d(num_features=64),
 13
 14             nn.Conv1d(in_channels=64, out_channels=128, kernel_size=101,
 15                       stride=20, padding=50),
 16             nn.Dropout(p=0.7),
 17             nn.LeakyReLU(negative_slope=0.2),
 18             nn.BatchNorm1d(num_features=128),
 19
 20             nn.Conv1d(in_channels=128, out_channels=256, kernel_size=101,
 21                       stride=20, padding=50),
 22             nn.Dropout(p=0.7),
 23             nn.LeakyReLU(negative_slope=0.2),
 24             nn.BatchNorm1d(num_features=256),
 25         )
 26
 27         self.convT_layers = nn.Sequential(
 28             nn.ConvTranspose1d(in_channels=256,
 29                                out_channels=128,
 30                                kernel_size=101,
 31                                stride=20,
 32                                padding=50,
 33                                output_padding=19),
 34             nn.Dropout(p=0.7),
 35             nn.LeakyReLU(negative_slope=0.2),
 36             nn.BatchNorm1d(num_features=128),
 37             nn.ConvTranspose1d(in_channels=128,
 38                                out_channels=64,
 39                                kernel_size=101,
 40                                stride=20,
 41                                padding=50,
 42                                output_padding=19),
 43             nn.Dropout(p=0.7),
 44             nn.LeakyReLU(negative_slope=0.2),
 45             nn.BatchNorm1d(num_features=64),
 46             nn.ConvTranspose1d(in_channels=64,
 47                                out_channels=1,
 48                                kernel_size=101,
 49                                stride=20,
 50                                padding=50,
 51                                output_padding=19),
 52         )
 53
 54     def forward(self, x):
 55         return self.convT_layers(self.conv_layers(x))
 56
 57 class mfcc_model(nn.Module):
 58     def __init__(self):
 59         super().__init__()
 60         self.conv_layers = nn.Sequential(
 61             nn.Conv2d(in_channels=1,
 62                       out_channels=64,
 63                       kernel_size=(20, 5),
 64                       stride=(10, 2),
 65                       padding=(10, 2)),
 66             nn.Dropout(p=0.7),
 67             nn.LeakyReLU(negative_slope=0.2),
 68             nn.BatchNorm2d(num_features=64),
 69             nn.Conv2d(in_channels=64,
 70                       out_channels=128,
 71                       kernel_size=(3, 3),
 72                       stride=(2, 2),
 73                       padding=(1, 1)),
 74             nn.Dropout(p=0.7),
 75             nn.LeakyReLU(negative_slope=0.2),
  1 import torch
  2 import torch.nn as nn
  3
  4 class baseline_model(nn.Module):
  5     def __init__(self):
  6         super().__init__()
  7         self.conv_layers = nn.Sequential(
  8             nn.Conv1d(in_channels=1, out_channels=64, kernel_size=101,
  9                       stride=20, padding=50),
 10             nn.Dropout(p=0.7),
 11             nn.LeakyReLU(negative_slope=0.2),
 12             nn.BatchNorm1d(num_features=64),
 13
 14             nn.Conv1d(in_channels=64, out_channels=128, kernel_size=101,
 15                       stride=20, padding=50),
 16             nn.Dropout(p=0.7),
 17             nn.LeakyReLU(negative_slope=0.2),
 18             nn.BatchNorm1d(num_features=128),
 19
 20             nn.Conv1d(in_channels=128, out_channels=256, kernel_size=101,
 21                       stride=20, padding=50),
 22             nn.Dropout(p=0.7),
 23             nn.LeakyReLU(negative_slope=0.2),
 24             nn.BatchNorm1d(num_features=256),
 25         )
 26
 27         self.convT_layers = nn.Sequential(
 28             nn.ConvTranspose1d(in_channels=256,
 29                                out_channels=128,
 30                                kernel_size=101,
 31                                stride=20,
 32                                padding=50,
 33                                output_padding=19),
 34             nn.Dropout(p=0.7),
 35             nn.LeakyReLU(negative_slope=0.2),
 36             nn.BatchNorm1d(num_features=128),
 37             nn.ConvTranspose1d(in_channels=128,
 38                                out_channels=64,
 39                                kernel_size=101,
 40                                stride=20,
 41                                padding=50,
 42                                output_padding=19),
 43             nn.Dropout(p=0.7),
 44             nn.LeakyReLU(negative_slope=0.2),
 45             nn.BatchNorm1d(num_features=64),
 46             nn.ConvTranspose1d(in_channels=64,
 47                                out_channels=1,
 48                                kernel_size=101,
 49                                stride=20,
 50                                padding=50,
 51                                output_padding=19),
 52         )
 53
 54     def forward(self, x):
 55         return self.convT_layers(self.conv_layers(x))
 56
 57 class mfcc_model(nn.Module):
 58     def __init__(self):
 59         super().__init__()
 60         self.conv_layers = nn.Sequential(
 61             nn.Conv2d(in_channels=1,
 62                       out_channels=64,
 63                       kernel_size=(20, 5),
 64                       stride=(10, 2),
 65                       padding=(10, 2)),
 66             nn.Dropout(p=0.7),
 67             nn.LeakyReLU(negative_slope=0.2),
 68             nn.BatchNorm2d(num_features=64),
 69             nn.Conv2d(in_channels=64,
 70                       out_channels=128,
 71                       kernel_size=(3, 3),
 72                       stride=(2, 2),
 73                       padding=(1, 1)),
 74             nn.Dropout(p=0.7),
 75             nn.LeakyReLU(negative_slope=0.2),
 76             nn.BatchNorm2d(num_features=128),
 77             nn.Conv2d(in_channels=128,
 78                       out_channels=256,
 79                       kernel_size=(13, 3),
 80                       stride=(1, 2),
 81                       padding=(0, 1)),
 82             nn.Dropout(p=0.7),
 83             nn.LeakyReLU(negative_slope=0.2),
 84             nn.BatchNorm2d(num_features=256),
 85         )
 86
 87         self.convT_layers = nn.Sequential(
 88             nn.ConvTranspose1d(in_channels=256,
 89                                out_channels=128,
 90                                kernel_size=101,
 91                                stride=20,
 92                                padding=50,
 93                                output_padding=19),
 94             nn.Dropout(p=0.7),
 95             nn.LeakyReLU(negative_slope=0.2),
 96             nn.BatchNorm1d(num_features=128),
 97             nn.ConvTranspose1d(in_channels=128,
 98                                out_channels=64,
 99                                kernel_size=101,
100                                stride=20,
101                                padding=50,
102                                output_padding=19),
103             nn.Dropout(p=0.7),
104             nn.LeakyReLU(negative_slope=0.2),
105             nn.BatchNorm1d(num_features=64),
106             nn.ConvTranspose1d(in_channels=64,
107                                out_channels=1,
108                                kernel_size=101,
109                                stride=10,
110                                padding=50,
  1 import torch
  2 import torch.nn as nn
  3
  4 class baseline_model(nn.Module):
  5     def __init__(self):
  6         super().__init__()
  7         self.conv_layers = nn.Sequential(
  8             nn.Conv1d(in_channels=1, out_channels=64, kernel_size=101,
  9                       stride=20, padding=50),
 10             nn.Dropout(p=0.7),
 11             nn.LeakyReLU(negative_slope=0.2),
 12             nn.BatchNorm1d(num_features=64),
 13
 14             nn.Conv1d(in_channels=64, out_channels=128, kernel_size=101,
 15                       stride=20, padding=50),
 16             nn.Dropout(p=0.7),
 17             nn.LeakyReLU(negative_slope=0.2),
 18             nn.BatchNorm1d(num_features=128),
 19
 20             nn.Conv1d(in_channels=128, out_channels=256, kernel_size=101,
 21                       stride=20, padding=50),
 22             nn.Dropout(p=0.7),
 23             nn.LeakyReLU(negative_slope=0.2),
 24             nn.BatchNorm1d(num_features=256),
 25         )
 26
 27         self.convT_layers = nn.Sequential(
 28             nn.ConvTranspose1d(in_channels=256,
 29                                out_channels=128,
 30                                kernel_size=101,
 31                                stride=20,
 32                                padding=50,
 33                                output_padding=19),
 34             nn.Dropout(p=0.7),
 35             nn.LeakyReLU(negative_slope=0.2),
 36             nn.BatchNorm1d(num_features=128),
 37             nn.ConvTranspose1d(in_channels=128,
 38                                out_channels=64,
 39                                kernel_size=101,
 40                                stride=20,
 41                                padding=50,
 42                                output_padding=19),
 43             nn.Dropout(p=0.7),
 44             nn.LeakyReLU(negative_slope=0.2),
 45             nn.BatchNorm1d(num_features=64),
 46             nn.ConvTranspose1d(in_channels=64,
 47                                out_channels=1,
 48                                kernel_size=101,
 49                                stride=20,
 50                                padding=50,
 51                                output_padding=19),
 52         )
 53
 54     def forward(self, x):
 55         return self.convT_layers(self.conv_layers(x))
 56
 57 class mfcc_model(nn.Module):
                                                                                                                                                                                                                            7,12       Наверху

-- ВИЗУАЛЬНАЯ СТРОКА --                                                                                                                                                                                                            180

▽
  1 import torch
 10             nn.Dropout(p=0.7),
 11             nn.LeakyReLU(negative_slope=0.2),
 38                                out_channels=64,
 39                                kernel_size=101,
 40                                stride=20,
 51                                output_padding=19),
 52         )
 53
 56
 57 class mfcc_model(nn.Module):
 58     def __init__(self):
 64                       stride=(10, 2),
 74             nn.Dropout(p=0.7),
 83             nn.LeakyReLU(negative_slope=0.2),
 84             nn.BatchNorm2d(num_features=256),
 99                                kernel_size=101,
100                                stride=20,
101                                padding=50,
102                                output_padding=19),
103             nn.Dropout(p=0.7),
107                                out_channels=1,
121         self.activation = nn.LeakyReLU(negative_slope=0.2)
129         self.conv1_batch_norm = nn.BatchNorm2d(num_features=64)
137
 28             nn.ConvTranspose1d(in_channels=256,
 29                                out_channels=128,
 30                                kernel_size=101,
 31                                stride=20,
 32                                padding=50,
 33                                output_padding=19),
 34             nn.Dropout(p=0.7),
 35             nn.LeakyReLU(negative_slope=0.2),
 36             nn.BatchNorm1d(num_features=128),
 37             nn.ConvTranspose1d(in_channels=128,
 38                                out_channels=64,
 39                                kernel_size=101,
 40                                stride=20,
 41                                padding=50,
 42                                output_padding=19),
 43             nn.Dropout(p=0.7),
 44             nn.LeakyReLU(negative_slope=0.2),
 45             nn.BatchNorm1d(num_features=64),
 46             nn.ConvTranspose1d(in_channels=64,
 47                                out_channels=1,
 48                                kernel_size=101,
 49                                stride=20,
 50                                padding=50,
 51                                output_padding=19),
 52         )
 53
 54     def forward(self, x):
 55         return self.convT_layers(self.conv_layers(x))
 56
 57 class mfcc_model(nn.Module):
 58     def __init__(self):
 59         super().__init__()
 60         self.conv_layers = nn.Sequential(
 61             nn.Conv2d(in_channels=1,
 62                       out_channels=64,
 63                       kernel_size=(20, 5),
 64                       stride=(10, 2),
 65                       padding=(10, 2)),
 66             nn.Dropout(p=0.7),
 67             nn.LeakyReLU(negative_slope=0.2),
 68             nn.BatchNorm2d(num_features=64),
 69             nn.Conv2d(in_channels=64,
 70                       out_channels=128,
 71                       kernel_size=(3, 3),
 72                       stride=(2, 2),
 73                       padding=(1, 1)),
 74             nn.Dropout(p=0.7),
 75             nn.LeakyReLU(negative_slope=0.2),
 76             nn.BatchNorm2d(num_features=128),
 77             nn.Conv2d(in_channels=128,
 78                       out_channels=256,
 79                       kernel_size=(13, 3),
 80                       stride=(1, 2),
 81                       padding=(0, 1)),
 82             nn.Dropout(p=0.7),
 83             nn.LeakyReLU(negative_slope=0.2),
 84             nn.BatchNorm2d(num_features=256),
segmentation.py                                                                                                                                                                                                             84,45          21%

