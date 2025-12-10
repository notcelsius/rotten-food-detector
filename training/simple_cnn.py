# -*- coding: utf-8 -*-
# Diego Escobar
import torch
import torchvision
import torchvision.transforms as transforms
import time

########################################################################
# HYPERPARAMETERS

EPOCH = 15
bs_arr = [16, 64]
lr_arr = [0.0003]
wd_arr = [0, 0.0001, 0.001]
momentum = 0.9

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

train_dir = "../data/train"
test_dir = "../data/test"

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def run_training(bs, lr, momentum, wd):

    trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                          shuffle=True, num_workers=0)
    testset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs,
                                         shuffle=False, num_workers=0)

    if __name__ == '__main__': # Windows multiprocessing workaround
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                                  shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=bs,
                                                 shuffle=False, num_workers=2)

    # Derive class names and number of classes from the dataset to avoid
    # mismatches between folder labels and the model output size.
    classes = trainset.classes
    NUM_CLASSES = len(classes)

    ########################################################################
    # Let us show some of the training images, for fun.

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import f1_score

    # functions to show an image


    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    # imshow(torchvision.utils.make_grid(images))
    # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


    ########################################################################
    # 2. Define a Convolution Neural Network
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Copy the neural network from the Neural Networks section before and modify it to
    # take 3-channel images (instead of 1-channel images as it was defined).

    import torch.nn as nn
    import torch.nn.functional as F


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            # Function nn.Conv2d(Input, Output, Kernel)
            # Input: 3 x 224 x 224
            # Conv1: (224 - 5 + 0) / 1 + 1 = 220
            # Result: 6 x 220 x 220
            self.conv1 = nn.Conv2d(3, 6, 5)

            # MaxPool2d reduces by factor of 2
            # Output: 6 x 110 x 110
            self.pool = nn.MaxPool2d(2, 2)

            # Conv2: (110 - 5 + 0) / 1 + 1 = 106
            # Result: 16 x 106 x 106
            self.conv2 = nn.Conv2d(6, 16, 5)

            # After pool2: 16 x 53 x 53
            # Fully connected layers
            self.fc1 = nn.Linear(16 * 53 * 53, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, NUM_CLASSES)

        def forward(self, x):
            # Conv1 + Pool: 6 x 110 x 110
            x = self.pool(F.relu(self.conv1(x)))
            # Conv2 + Pool: 16 x 53 x 53
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 53 * 53)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = Net()

    ########################################################################
    # 3. Define a Loss function and optimizer
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Let's use a Classification Cross-Entropy loss and SGD with momentum.

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)

    ########################################################################
    # 4. Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    #
    # This is when things start to get interesting.
    # We simply have to loop over our data iterator, and feed the inputs to the
    # network and optimize.

    loss_per_epoch = []
    train_acc_per_epoch = []
    test_acc_per_epoch = []
    f1_per_epoch = []
    best_f1 = -1.0

    start_time = time.time()

    for epoch in range(EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        train_corrects = 0
        train_total = 0

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels)
            train_total += labels.size(0)

        preds_list = []
        labels_list = []
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                preds_list.append(predicted.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        loss_per_epoch.append(running_loss / train_total)
        # train_acc_per_epoch.append(100.0 * train_corrects / train_total)
        # test_acc_per_epoch.append(100.0 * correct / total)

        # compute macro F1 using scikit-learn
        if len(preds_list) > 0:
            preds_all = np.concatenate(preds_list)
            labels_all = np.concatenate(labels_list)
            test_f1 = f1_score(labels_all, preds_all, average='macro')
        else:
            test_f1 = 0.0

        f1_per_epoch.append(100.0 * test_f1)

        print("  Epoch: %d/%d | train_loss=%.4f train_acc=%.2f%% test_acc=%.2f%% | f1=%.4f" % (
            epoch + 1,
            EPOCH,
            running_loss / train_total,
            100.0 * train_corrects / train_total,
            100.0 * correct / total,
            test_f1
        ))

        # Save best checkpoint by macro-F1 (raw value)
        if test_f1 > best_f1:
            best_f1 = test_f1
            BEST_PATH = "final_simple_cnn/" + f".conv2.bs{bs}.lr{lr}.wd{wd}" + ".best.pt"
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item() if 'loss' in locals() else None,
                    'classes': classes,
                    'num_classes': NUM_CLASSES,
                    'best_f1': best_f1,
                }, BEST_PATH)
                # print(f"Saved new best checkpoint (F1={best_f1:.4f}) to: {BEST_PATH}")
            except Exception as e:
                print(f"Warning: could not save best checkpoint: {e}")

    end_time = time.time()
    elapsed = end_time - start_time
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)

    print("---------------------")
    print("Best F1 Score: %.4f" % best_f1)
    print(f"Total training time: {int(h)}:{int(m):02d}:{int(s):02d} (hh:mm:ss) — {elapsed:.2f} seconds")
    print("---------------------")
    print(" ")

if (__name__ == '__main__'):
    for bs in bs_arr:
        for lr in lr_arr:
            for wd in wd_arr:
                print("Training with: conv=2, epoch=%d, lr=%.4f, momentum=%.2f, weight_decay=%.4f, batch_size=%d" %
                    (EPOCH, lr, momentum, wd, bs))
                run_training(bs, lr, momentum, wd)