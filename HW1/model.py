import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



if torch.cuda.is_available():
    device = torch.device("cuda")
    
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU")

transform = transforms.Compose([
    
    transforms.RandomCrop(32, padding=4),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
   transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])
transform2 = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])


trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = CIFAR10(root='./data', train=False, download=True, transform=transform2)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


model = models.vgg19_bn(num_classes=10) 
model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


num_epochs = 60
best_accuracy = 0.0
train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    step=0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        model.cuda()
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        step+=1

    train_accuracy = 100 * correct_train / total_train
    res = running_loss/step
    train_losses.append(running_loss/step)
    train_accuracies.append(train_accuracy)


    
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    step_2=0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            step_2+=1

    test_accuracy = 100 * correct_test / total_test
    test_losses.append(test_loss/step_2)
    test_accuracies.append(test_accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss/step:.4f}, Train Acc: {train_accuracy:.2f}%, Test Loss: {test_loss/step_2:.4f}, Test Acc: {test_accuracy:.2f}%")


    
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), "best_model.pth")


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")

plt.savefig("training_curve.png")
plt.show()
