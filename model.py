import torch


class ClassifierModel(torch.nn.Module):
    def __init__(self):
        super(ClassifierModel, self).__init__()

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(
                in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=1
            ),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(
                in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1
            ),
            torch.nn.BatchNorm2d(16),
            torch.nn.Flatten(),
        )
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(256, 50),
            torch.nn.ReLU(True),
            torch.nn.Linear(50, 10),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.backbone(x)
        y = self.proj(x)
        return y
