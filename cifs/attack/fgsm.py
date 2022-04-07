import torch
import torch.nn as nn
from torch import optim


"""
ϵ（epsilon）的值通常是人为设定 ，可以理解为学习率，一旦扰动值超出阈值，该对抗样本会被人眼识别。
"""
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fgsm_attack(model,images,labels,eps):
    with torch.enable_grad():
        criterion = nn.CrossEntropyLoss()
        images = images.to(device)
        labels = labels.to(device)
        # 设置张量的requires_grad属性，这对于攻击很关键
        images.requires_grad = True
        # 通过模型前向传递数据
        outputs,_ = model(images)
        # _,init_pred = outputs.max(1, keepdim=True)  # get the index of the max log-probability
        # # 如果初始预测是错误的，不打断攻击，继续
        # if init_pred.item() != labels.item():
        #     print("如果初始预测是错误的，不打断攻击，继续")
        # 计算损失
        loss = criterion(outputs, labels)
        # 将所有现有的渐变归零
        model.zero_grad()
        # 计算后向传递模型的梯度
        loss.backward()
        # 收集datagrad
        data_grad = images.grad.data
        # 使用sign（符号）函数，将对x求了偏导的梯度进行符号化
        sign_data_grad = data_grad.sign()
        # 通过epsilon生成对抗样本
        perturbed_image = images + eps * sign_data_grad
        # 做一个剪裁的工作，将torch.clamp内部大于1的数值变为1，小于0的数值等于0，防止image越界
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # 返回对抗样本
        return perturbed_image
