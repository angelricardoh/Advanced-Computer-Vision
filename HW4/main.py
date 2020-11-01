import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import model as model
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # use GPU if available
PATH = './cifar_net.pth'

def train(trainloader, net, criterion, optimizer):
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # access data
            # inputs, labels = data 
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            # forward
            outputs = net(inputs)

            # loss
            loss = criterion(outputs, labels)

            # backward
            loss.backward()

            # update the weights
            optimizer.step() # 1 step over SGD

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # printing every 2000 mini-batches/iterations
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100)) # last term is the average of running loss
            running_loss = 0.0

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_and_check_results(net, images, testloader):
    net.load_state_dict(torch.load(PATH))

    outputs = net(images)
    print(outputs)
    print(outputs.shape)

    max_val, predicted = torch.max(outputs, 1)    # torch.max returns the maximum value as well as the index
    # second argument stands for the dimension I want to compute the max value of
    print(max_val, predicted)

    # Print the predicted classes
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                for j in range(4)))

                                # Loop over the test set and compute accuracy
    correct = 0
    total = 0

    # I don't want to compute grad during my test time
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            outputs = net(images)

            max_val, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item() # item gives you the numerical value of that

    print('Accuracy: ', 100 * correct / total)

if __name__ == '__main__':
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)


    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)


    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=80,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=80,
                                            shuffle=False, num_workers=2)

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # print(images.size())

    # # show images
    # imshow(torchvision.utils.make_grid(images))

    # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    net = model.Net()

    # Run training

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train(trainloader, net, criterion, optimizer)

    # torch.save(net.state_dict(), PATH)

    # Show Results
    load_and_check_results(net, images, testloader)

