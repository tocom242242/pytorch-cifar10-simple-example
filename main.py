import numpy as np
import torch
from tqdm import tqdm

from data import get_loaders
from model import ClassifierModel
from utils import AverageMeter

EPOCHS = 1
BATCH_SIZE = 32

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders(batch_size=BATCH_SIZE)
    # plot_ds(train_loader.dataset)
    model = ClassifierModel()
    train_loss = AverageMeter("train_loss")
    train_acc = AverageMeter("train_acc")
    val_loss = AverageMeter("val_loss")
    val_acc = AverageMeter("val_acc")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    min_loss = np.inf
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        for x, y in train_loader:
            outputs = model(x)
            loss = criterion(outputs.squeeze(), y)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y).sum().item() / y.size(0)
            train_loss.update(loss.data)
            train_acc.update(accuracy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        for x, y in val_loader:
            outputs = model(x)
            loss = criterion(outputs.squeeze(), y)
            val_loss.update(loss.data)

            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y).sum().item() / y.size(0)
            val_acc.update(accuracy)

        if val_loss.avg < min_loss:
            torch.save(model.state_dict(), "model.pth")
            min_loss = val_loss.avg

        print(
            "[epoch :{:.1f} train_loss: {} val_loss: {} train_acc: {} val_acc: {}] ".format(
                epoch,
                train_loss.avg,
                val_loss.avg,
                train_acc.avg,
                val_acc.avg,
            )
        )
        scheduler.step()
        train_loss.reset()
        val_loss.reset()

    model.load_state_dict(torch.load("model.pth"))

    model.eval()
    accuracy = 0
    num_total = 0
    for x, y in test_loader:
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        accuracy += (predicted == y).sum().item()
        num_total += y.size(0)

    accracy = accuracy / num_total
    print(f"accuracy:{accracy}")
