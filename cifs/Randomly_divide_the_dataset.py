import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset


def Randomly_divide_the_dataset(BACTHSIZE,name):
  # 测试集随机划分数据集
  trainset_transform = transforms.Compose([
      transforms.ToTensor(),
      # 归一化处理
      # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
  ])
  testset_transform = transforms.Compose([
      transforms.ToTensor(),
      # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  data_train = torchvision.datasets.CIFAR10(root='./data/cifar10/',
                                            train=True,
                                            transform=trainset_transform,
                                            download=False)
  data_test = torchvision.datasets.CIFAR10(root='./data/cifar10/',
                                           train=False,
                                           transform=testset_transform,
                                           download=False)
  indices = []
  idx = data_test.class_to_idx[name]

  for i in range(len(data_test)):
    current_class = data_test[i][1]
    if current_class == idx:
      indices.append(i)
  indices = indices[:int(0.5 * len(indices))]
  new_dataset = Subset(data_test, indices)
  new_data_loader_test = DataLoader(dataset=new_dataset,
                                    batch_size=BACTHSIZE,
                                    shuffle=False,
                                    )
  return new_data_loader_test

def newData(BATCHSIZE,name):
    return Randomly_divide_the_dataset(BACTHSIZE=BATCHSIZE,name=name)

if __name__ == "__main__":
    name = 'truck'
    newData(BATCHSIZE=10,name=name)