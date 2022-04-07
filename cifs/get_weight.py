import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from Randomly_divide_the_dataset import newData
from models.resnet import ResNet18
from metric import *
from attack.pgdattack import _pgd_whitebox

import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'



models_path = './modelsaved/resnet18_050_20220322.pth'
# models_path = './modelsaved/v2_pgd_adversarial_epoch50.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = ResNet18().to(device)
net.load_state_dict(torch.load(models_path,map_location="cuda:0"))

# 打印网络结构
for param in net.named_parameters():
    print(param[1].size())
    print(param[0])

# 打印最后一层linear的权重输出

BATCHSIZE = 1

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
data_loader = newData(BATCHSIZE=BATCHSIZE,name=classes[0])
loss_func = nn.CrossEntropyLoss()

# 定义hook

class Hook():
    def __init__(self,module,backward=True):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self,module,input,output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

# 求一张图片的某一层 top-k logits之和的梯度
# 定义top-k logits函数
# 返回的是一张图片输出最大的两个值
def predict_from_logits_topk(logits,dim=1,topk=5):
    """
    :param logits:
    :param dim:
    :param topk:
    :return: logits.topk(topk,dim)[0]
    """
    # print(logits.topk(topk,dim))
    # torch.return_types.topk(
    # values=tensor([[1.7568, 0.7012]], device='cuda:0'),
    # indices=tensor([[3, 2]], device='cuda:0'))

    # print(logits.topk(topk,dim)[0])
    # tensor([[1.7568, 0.7012]], device='cuda:0')

    # print(logits.topk(topk,dim)[1])
    # tensor([[3, 2]], device='cuda:0')
    return logits.topk(topk,dim)[0]


# output_result = [] # 这两个其实是用不到的
# output_grad = []
input_result = []
input_grad = []

hookB = Hook(net.linear, backward=True) # 返回梯度信息
hookF = Hook(net.linear,backward=False) # 返回中间层特征
grad_magnitude = 0.0
for index,(inputs,labels) in enumerate(data_loader):
    # output_result.clear()
    # output_grad.clear()
    input_result.clear()
    input_grad.clear()

    inputs,labels = inputs.to(device),labels.to(device)
    inputs = _pgd_whitebox(net, inputs, labels)

    outputs,outputs1 = net(inputs)

    print(outputs)
    # tensor([[ 0.2797, -0.4184,  0.7012,  1.7568,  0.0703,  0.5413, -0.5254, -0.4241,
    #          -0.3479,  0.0046]], device='cuda:0')

    print(predict_from_logits_topk(outputs,topk=2))
    # tensor([[1.7568, 0.7012]], device='cuda:0')
    topk_outputs = predict_from_logits_topk(outputs,dim=1,topk=2)
    sum_topk_outputs = sum(topk_outputs[0])
    print(sum_topk_outputs)
    # (sum_topk_outputs).backward() # top-2输出logits之和求导
    (topk_outputs[0][0]+topk_outputs[0][1]).backward() # top-2输出logits之和求导
    input_grad.append(hookB.input) # 将top-5输出logits之和之梯度返回到列表中
    # 接下来我要对每张图片的top-5输出做处理

    target = input_grad[0][1].cpu().detach().numpy()
    print(target)
    if index == 0 :
        grad_magnitude = target
        # print(grad_magnitude)
    else:
        grad_magnitude = (grad_magnitude+target)/2.
        # print(grad_magnitude)

print(grad_magnitude)
plt.plot(np.sort(grad_magnitude[0]))
plt.grid()
plt.show()









#
# clncorrect = 0
# idx_batch = 0
# num_examples = 0
#
# lst_label = []
# lst_pred = []
#
# for clndata, target in tqdm(data_loader):
#     clndata, target = clndata.to(device), target.to(device)
#     # with torch.no_grad():
#     output = net(clndata)
#     pred = predict_from_logits_topk(output, topk=5)
#     print(pred)
#     lst_label.append(target)
#     lst_pred.append(pred)
#
#     num_examples += clndata.shape[0]
#     idx_batch += 1
#     if idx_batch == BATCHSIZE:
#         break
#
# label = torch.cat(lst_label).view(-1, 1)
# print(label)
# pred = torch.cat(lst_pred).view(-1, 5)
# print(pred)
# num = label.size(0)
# print(num)
# accuracy = (label == pred).sum().item() / num
# print(accuracy)
#
# message = '***** Test set acc: ({:.2f}%)'.format(100. * accuracy)
# print(message)













