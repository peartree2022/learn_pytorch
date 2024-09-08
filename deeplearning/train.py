import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from parameter import *
from Resnet import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder(root='./eyesdataset/train',transform=train_transforms)
test_dataset = datasets.ImageFolder(root='./eyesdataset/val', transform=test_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_worker)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_worker)

def train(model, train_dataloader, criterion, optimizer, epoches):
    best_acc = 0
    for epoch in range(epoches):
        model.train()
        running_loss = 0
        for inputs, targets in tqdm(train_dataloader, desc=f'epoch:{epoch + 1} / {epoches}', unit='batch'):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataloader.dataset)
        print(f'epoch[{epoch + 1} / {epoches}, train_loss{epoch_loss: .4f}]')
        evaluate(model, test_dataloader, criterion)
        save(model, save_path, epoch)

def evaluate(model, test_dataloader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = criterion(output, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(output, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    avg_loss = test_loss / len(test_dataloader.dataset)
    accuracy = 100 * correct / total
    print(f'test loss:{avg_loss: .4f}, Accuracy:{accuracy: .2f}%')
    return accuracy

def save(model, path, epoch):
    torch.save(model.state_dict(), save_path / f'model-{epoch}.pth')


if __name__ == '__main__':
    model = Resnet18(num_class)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(model, train_dataloader, criterion, optimizer, epoches)
