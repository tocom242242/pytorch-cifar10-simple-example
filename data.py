import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import InterpolationMode

BICUBIC = InterpolationMode.BICUBIC


def get_loaders(batch_size):
    ds = torchvision.datasets.CIFAR10
    transform = transforms.Compose(
        [
            transforms.Resize(32, interpolation=BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    trainset = ds(root="data", train=True, download=True, transform=transform)
    indices = torch.arange(10000)
    trainset = Subset(trainset, indices)

    n_samples = len(trainset)
    train_size = int(len(trainset) * 0.9)
    val_size = n_samples - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False
    )
    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False
    )

    testset = ds(root="data", train=False, download=True, transform=transform)
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False
    )
    return train_loader, val_loader, test_loader


def plot_ds(dataset, row=10, col=1, figsize=(20, 10)):
    fig_img, ax_img = plt.subplots(row, col, figsize=figsize, tight_layout=True)
    plt.figure()
    for i in range(row):
        img1, _ = dataset[i]
        img1 = denormalization(img1)
        img1 = np.squeeze(img1)
        ax_img[i].imshow(img1)

    fig_img.savefig("data_sample.png", dpi=100)
    plt.close()


def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def denormalization(x):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    x = inverse_normalize(x, mean, std)
    x = x.cpu().detach().numpy()
    # x = (x.transpose(1, 2, 0)).astype(np.uint8)
    x = (x.transpose(1, 2, 0) * 255.0).astype(np.uint8)

    return x
