import torch
import torch.nn as nn
import os
import numpy as np
from attack.pgdattack import _pgd_whitebox
from attack.fgsm import fgsm_attack
from models.resnet import ResNet18
from Randomly_divide_the_dataset import newData
from torch.autograd import Variable
import torch.optim as optim

BATCHSIZE = 64
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
for index in range(10):
    test_data_loader=newData(BATCHSIZE=BATCHSIZE,name=classes[index])
    # model_path = './modelsaved/v2_standard_model_epoch50.pth'
    model_path = './modelsaved/v2_pgd_adversarial_epoch50.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_std = ResNet18().to(device)
    net_adv = ResNet18().to(device)
    net = ResNet18().to(device)  # 用来生成对抗训练的测试集图片
    net_std.load_state_dict(torch.load(model_path,map_location='cuda:0'))
    net_adv.load_state_dict(torch.load(model_path,map_location='cuda:0'))
    net.load_state_dict(torch.load(model_path,map_location='cuda:0'))
    output_for_layer4_for_net_std = []
    output_for_layer4_for_net_adv = []
    def get_output_in_layer4_for_std_in_std_trainning(module,data_input,data_output):
        output_for_layer4_for_net_std.append(data_output)
    def get_output_in_layer4_for_adv_in_std_trainning(module,data_input,data_output):
        output_for_layer4_for_net_adv.append(data_output)


    with torch.no_grad():
        relu_index = 0
        net_std.layer4.register_forward_hook(get_output_in_layer4_for_std_in_std_trainning)
        net_adv.layer4.register_forward_hook(get_output_in_layer4_for_adv_in_std_trainning)
        statis_results_std=0.0
        magnitude_std=0.0
        statis_results_adv=0.0
        magnitude_adv=0.0
        BATCH_COUNTERS = 0
        SAMPLES_COUNTERS = 0

    #     我需要的是magnitudes和counters
        for datas,labels in test_data_loader:
            output_for_layer4_for_net_std.clear()
            output_for_layer4_for_net_adv.clear()
            datas,labels = datas.to(device),labels.to(device)
            # 生成对抗训练集
            # advdatas = _pgd_whitebox(net,datas,labels)
            advdatas = fgsm_attack(net, datas, labels, 0.03)
            outputs1,_ = net_std(datas, _eval=True)
            # print(outputs1)
            outputs2,_ = net_adv(advdatas, _eval=True)
            # print(outputs2)
            pred1 = torch.max(outputs1, dim=1)[1]
            pred2 = torch.max(outputs2, dim=1)[1]
            # print(" 当前正常样本预测标签：",pred1,'\n',"当前对抗样本预测标签",pred2)
            #################################################################
            #################################################################
            idx = np.where(labels.cpu().numpy() == np.array([index]*datas.shape[0]))[0]
            # print(idx)
            #################################################################
            #################################################################
            idx = torch.tensor(idx)
            SAMPLES_COUNTERS += len(idx)

            if(len(idx)>0):
                feat_out = output_for_layer4_for_net_std[0][idx]
                if len(feat_out.shape) == 4:
                    N, C, H, W = feat_out.shape
                    feat_out = feat_out.view(N, C, H * W)
                    feat_out = torch.mean(feat_out, dim=-1)#返回格式为（N,C,1）or（N，C）

                N,C =feat_out.shape
                max_value = (torch.max(feat_out,dim=0,keepdim=True)[0])
                threshold = 1e-2 *max_value
                # print(max_value)
                mask = feat_out>threshold.expand(N,C)
                # feat_out = feat_out*mask
                for c in range(C):
                    for n in range(N):
                        feat_out[n][c] = feat_out[n][c]/abs(max_value[0][c])
                COUNTER_ACTIVATE = torch.sum(mask,dim=0).view(C)
                feat_mean_magnitude = torch.sum(feat_out,dim=0).view(C)
                for k in range(C):
                    if feat_mean_magnitude[k] != 0:
                        feat_mean_magnitude[k] = feat_mean_magnitude[k] / (COUNTER_ACTIVATE[k].float())
                COUNTER_ACTIVATE = COUNTER_ACTIVATE.cpu().numpy()
                feat_mean_magnitude =  feat_mean_magnitude.cpu().numpy()
                if BATCH_COUNTERS == 0:
                    statis_results_std = COUNTER_ACTIVATE
                    magnitude_std = feat_mean_magnitude
                else:
                    statis_results_std = statis_results_std + COUNTER_ACTIVATE
                    magnitude_std = (magnitude_std + feat_mean_magnitude) / 2
            if (len(idx) > 0):
                feat_out = output_for_layer4_for_net_adv[0][idx]
                if len(feat_out.shape) == 4:
                    N, C, H, W = feat_out.shape
                    feat_out = feat_out.view(N, C, H * W)
                    feat_out = torch.mean(feat_out, dim=-1)  # 返回格式为（N,C,1）or（N，C）
                N, C = feat_out.shape
                max_value = (torch.max(feat_out,dim=0,keepdim=True)[0])
                threshold = 1e-2 * max_value
                mask = feat_out > threshold.expand(N, C)
                # feat_out = feat_out * mask
                for c in range(C):
                    for n in range(N):
                        feat_out[n][c] = feat_out[n][c] / abs(max_value[0][c])
                COUNTER_ACTIVATE = torch.sum(mask, dim=0).view(C)
                feat_mean_magnitude = torch.sum(feat_out, dim=0).view(C)
                for k in range(C):
                    if feat_mean_magnitude[k] != 0:
                        feat_mean_magnitude[k] = feat_mean_magnitude[k] / (COUNTER_ACTIVATE[k].float())
                COUNTER_ACTIVATE = COUNTER_ACTIVATE.cpu().numpy()
                feat_mean_magnitude = feat_mean_magnitude.cpu().numpy()
                if BATCH_COUNTERS == 0:
                    statis_results_adv = COUNTER_ACTIVATE
                    magnitude_adv = feat_mean_magnitude
                else:
                    statis_results_adv = statis_results_adv + COUNTER_ACTIVATE
                    magnitude_adv = (magnitude_adv + feat_mean_magnitude) / 2

            BATCH_COUNTERS += 1

        print('Count Samples', SAMPLES_COUNTERS)
        statis_results_adv = np.array(statis_results_adv)
        statis_results_std = np.array(statis_results_std)
        print(statis_results_std.shape)
        print(statis_results_adv.shape)
        res = np.concatenate([statis_results_adv, statis_results_std], axis=0)
        if os.path.exists('./Frequency') == False:
            os.makedirs('./Frequency')
        np.save('Frequency/cifar10_pgd_training_layer4_class{}_1e-2_1_for_fgsm.npy'.format(index), res)

        magnitude_results_adv = np.array(magnitude_adv)
        magnitude_results_std = np.array(magnitude_std)
        res = np.concatenate([magnitude_results_adv, magnitude_results_std], axis=0)
        if os.path.exists('./Magnitude') == False:
            os.makedirs('./Magnitude')
        np.save('Magnitude/cifar10_pgd_training_layer4_class{}_1e-2_1_for_fgsm.npy'.format(index), res)












