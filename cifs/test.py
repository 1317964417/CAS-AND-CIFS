import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from models.resnet import ResNet18
from Randomly_divide_the_dataset import newData
from cifar10 import cifar10_data_download
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ResNet18().to(device)
# net.load_state_dict(torch.load("./modelsaved/standard.pth",map_location="cuda:0"))
net.load_state_dict(torch.load("./modelsaved/v2_standard_model_epoch50.pth",map_location="cuda:0"))
# net.load_state_dict(torch.load("./modelsaved/v2_pgd_adversarial_epoch50.pth",map_location="cuda:0"))

EPOCH = 8
BATCHSIZE = 64
LR = 0.005

#
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.8, weight_decay=4e-4)
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
data_loader = newData(BATCHSIZE=1,name=classes[0])
data_loader_train, data_loader_test = cifar10_data_download(BATCHSIZE)


def pgd_attack(model, images, labels, eps, alpha=2 / 255, iters=40):
    # 将输入和标签转换为GPU可用的类型
    images = images.to(device)
    labels = labels.to(device)
    loss = criterion

    ori_images  = images.data
    for i in range(iters):
        images.requires_grad =  True
        outputs,_ = model(images)
        model.zero_grad()
        cost = loss(outputs,labels).to(device)
        cost.backward()
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
    return images

def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()

def tes():
    print("True Image & Predicted Label")

    net.eval()

    correct = 0
    total = 0

    for images, labels in data_loader_test:
        images = images.to(device)
        labels = labels.to(device)
        images = pgd_attack(net, images, labels, eps=0.03, alpha= 0.003, iters=20)
        outputs,_ = net(images, _eval=True)
        # print(outputs)
        _, pre = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (pre == labels).sum()

        # imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True),[classes[i]for i in pre])

    print('Accuracy of test text: %f %%' % (100 * float(correct) / total))


if __name__ == "__main__":

    tes()





