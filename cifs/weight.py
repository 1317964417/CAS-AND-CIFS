import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from Randomly_divide_the_dataset import newData
from models.resnet import ResNet18
from models.resnetforcifs import ResNet18_1
# from metric import *
from attack.pgdattack import _pgd_whitebox
from attack.fgsm import fgsm_attack
from attack.cw import  cw_attack

import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'



models_path = './modelsaved/v2_standard_model_epoch50.pth'
# models_path = './modelsaved/v2_pgd_adversarial_epoch50.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = ResNet18().to(device)
net_1 = ResNet18_1().to(device)
net_adv = ResNet18().to(device)
net_std = ResNet18().to(device)

net.load_state_dict(torch.load(models_path,map_location="cuda:0"))
net_1.load_state_dict(torch.load(models_path,map_location="cuda:0"))
net_adv.load_state_dict(torch.load(models_path,map_location="cuda:0"))
net_std.load_state_dict(torch.load(models_path,map_location="cuda:0"))



# 打印网络结构
for param in net.named_parameters():
    print(param[1].size())
    print(param[0])

# 打印最后一层linear的权重输出

BATCHSIZE = 1
for kind in range(10):
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    data_loader = newData(BATCHSIZE=BATCHSIZE,name=classes[kind])

    # 定义hook钩子保存中间变量
    output_grad = []
    def get_grad(grad):
        output_grad.append(grad)

    # 求一张图片的某一层 top-k logits之和的梯度
    # 定义top-k logits函数
    # 返回的是一张图片输出最大的两个值
    def predict_from_logits_topk(logits,dim=1,topk=2):
        """
        :param logits:
        :param dim:
        :param topk:
        :return: logits.topk(topk,dim)[0]
        """
        return logits.topk(topk,dim)[0]
    # 此处实现的是一张图片在某一层的top-k的logits输出之和求导之结果
    # for index,(inputs,labels) in enumerate(data_loader):
    #     output_grad.clear()
    #     inputs,labels = inputs.to(device),labels.to(device)
    #     _,outputs1 = net(inputs)
    #     print(outputs1)
    #     outputs = net.linear(outputs1)
    #     print(outputs)
    #     topk_outputs = predict_from_logits_topk(outputs, dim=1, topk=2)
    #     print(topk_outputs)
    #     outputs1.register_hook(get_grad)
    #     sum_topk_outputs = sum(topk_outputs[0])
    #     sum_topk_outputs.backward()
    # print(output_grad[0][0].cpu().detach().numpy())
    # result = output_grad[0][0].cpu().detach().numpy()
    # plt.plot(np.sort(result))
    # plt.grid()
    # plt.show()

    # 多张图片某一层top-k的logits输出之和求梯度



    net.eval()
    net_1.eval()
    net_std.eval()
    net_adv.eval()

    weight_magnitude = 0.0
    for index,(inputs,labels) in enumerate(data_loader):
        output_grad.clear()
        inputs,labels = inputs.to(device),labels.to(device)
        inputs = _pgd_whitebox(net, inputs, labels)
        # inputs = fgsm_attack(net,inputs,labels,0.03)
        # inputs = cw_attack(net_1,inputs,labels,BATCHSIZE=1)
        _,outputs1 = net(inputs, _eval=True)  # outputs1 为最后一层线性层的输入，对应的是Z^l，按照论文的要求可知其维度是512
        # print(result)
        with torch.enable_grad():
            outputs1.requires_grad_(True)
            # print(outputs1)
            outputs = net_std.linear(outputs1) # outputs 为最后一层线性层的输出，对应的是最后A^l(Z^l)
            # print(outputs)
            topk_outputs = predict_from_logits_topk(outputs,dim=1,topk=3)
            outputs1.register_hook(get_grad)  # 注册钩子，用来勾取所谓的作为的中间变量从而未保存的梯度
            sum_topk_outputs = topk_outputs[0][0]+topk_outputs[0][1]+topk_outputs[0][2]
            sum_topk_outputs.backward()
            result = output_grad[0][0].cpu().detach().numpy()
            if index == 0:
                weight_magnitude = result
            else:
                weight_magnitude = (weight_magnitude+result)/2

        print("________")

    weight_magnitude = weight_magnitude
    # print('图片数量：',index+1)
    weight  = np.array(weight_magnitude)
    # print(weight.shape)
    flag0 = ['natural', 'pgd_10', 'fgsm', 'cw']
    flag1 = ['std','adv']
    saved_path = "./weight/{}/{}/{}".format(flag0[3],flag1[0],classes[kind])

    if os.path.exists(saved_path) == False:
        os.makedirs(saved_path)
    np.save(saved_path+'/'+'cifar10_'+classes[kind]+'_weight',weight)
    weight = np.sort(weight)[::-1]
    # weight=abs(np.sort(weight))
    fig = plt.figure(figsize=(10, 9))
    left, bottom, width, height = 0.15, 0.15, 0.8, 0.8
    ax1 = fig.add_axes([left, bottom, width, height])
    ax1.set_ylim(-1, 1)
    ax1.set_xlim(0,512)
    xaxis = range(len(weight))
    ax1.bar(xaxis,weight,color='orange',alpha=1,label='weight')
    ax1.set_xlabel('Channel', fontdict={'family': 'Times Roman', 'size': 12})
    ax1.set_ylabel('Magnitude of weight', fontdict={'family': 'Times Roman', 'size': 12})
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], labels[::-1], loc=0, ncol=1, prop={'family': 'Times Roman', 'size': 24})
    ax1.tick_params(axis='both', which='major', labelsize=24)
    ax1.tick_params(axis='both', which='minor', labelsize=24)
    plt.grid()

    if os.path.exists(saved_path) == False:
        os.makedirs(saved_path)
    plt.savefig(saved_path+'/'+'_picture.png')
















































