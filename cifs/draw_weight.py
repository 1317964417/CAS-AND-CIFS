import torch
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 实现画图
# 用干净样本训练出来的模型的weight（干净样本和对抗样本）、激活幅度（干净样本和对抗样本）、激活数量（干净样本和对抗样本）
# 用对抗样本训练出来的模型（鲁棒性）的weight（干净样本和对抗样本）、激活幅度（干净样本和对抗样本）、激活数量（干净样本和对抗样本）
# 总共有10个类别，至于对抗样本，由fgsm、pgd-20、cw生成

# 第一个图，需要画干净样本训练出来的模型再不同测试集下的weight（干净样本）
# 第二个图，需要画干净样本训练出来的模型再不同测试集下的weight（pgd10对抗样本）
for index in range(10):
    plt.figure(index+1)
    plt.grid(linestyle='--')
    testdata=('natural','fgsm','pgd_10','CW')
    training_model = ('std','adv')
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    natural_data_weight_path_std = './weight/{}/{}/{}/cifar10_{}_weight.npy'.format(testdata[0],training_model[0],classes[index],classes[index])
    natural_data_weight_path_adv = './weight/{}/{}/{}/cifar10_{}_weight.npy'.format(testdata[1],training_model[0],classes[index],classes[index])
    weight_std=np.load(natural_data_weight_path_std)
    weight_adv=np.load(natural_data_weight_path_adv)
    saved_path = './picture/weight/fgsm_and_natural_based_std_training/{}/'.format(classes[index])
    xais0 =range(len(weight_std))
    xais1 =range(len(weight_adv))
    # 准备画布
    fig = plt.subplot(1,1,1)
    fig.set_ylim(-1,1)
    fig.set_xlim(0.0,512.0)
    weight_std = np.sort(weight_std)[::-1]
    weight_adv = np.sort(weight_adv)[::-1]
    fig.bar(xais0, weight_std, color='blue',alpha=0.7,label='{}_std_training_natural examples'.format(classes[index]))
    fig.bar(xais1, weight_adv, color='orange',alpha=0.5,label='{}_std_training_fgsm examples'.format(classes[index]))
    fig.set_xlabel("channel",fontdict={'family': 'Times Roman', 'size': 12})
    fig.set_ylabel("weight",fontdict={'family': 'Times Roman', 'size': 12})
    handles, labels = fig.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc=0, ncol=1, prop={'family': 'Times Roman', 'size': 12})
    fig.tick_params(axis='both', which='major', labelsize=12)
    fig.tick_params(axis='both', which='minor', labelsize=12)
    if os.path.exists(saved_path) == False:
        os.makedirs(saved_path)
    plt.savefig(saved_path+'{}.png'.format(classes[index]))
plt.grid(linestyle='--')
# plt.figure(2)
# for index in range(10):
#     plt.grid(linestyle='--')
#     testdata=('natural','FGSM','PGD','CW')
#     training_model = ('std','std')
#     classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#     natural_data_weight_path = './weight/{}/{}/{}/cifar10_{}_weight.npy'.format(testdata[0],training_model[0],classes[index],classes[index])
#     weight=np.load(natural_data_weight_path)
#     saved_path = './picture/weight/std_training/natural/'
#     xais =range(len(weight))
#     # 准备画布
#     fig = plt.subplot(1,1,1)
#     fig.set_ylim(-1,1)
#     fig.set_xlim(0.0,512.0)
#     weight = np.sort(weight)[::-1]
#     fig.bar(xais, weight,alpha=0.5, label='{}_natural examples'.format(classes[index]))
#     fig.set_xlabel("channel",fontdict={'family': 'Times Roman', 'size': 12})
#     fig.set_ylabel("weight",fontdict={'family': 'Times Roman', 'size': 12})
#     handles, labels = fig.get_legend_handles_labels()
#     fig.legend(handles[::-1], labels[::-1], loc=0, ncol=1, prop={'family': 'Times Roman', 'size': 12})
#     fig.tick_params(axis='both', which='major', labelsize=12)
#     fig.tick_params(axis='both', which='minor', labelsize=12)
# plt.grid(linestyle='--')
plt.show()









