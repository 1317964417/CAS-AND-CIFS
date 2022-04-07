import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
for index in range(10):
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    weight_file_path_for_natural_std_training = './weight/natural/adv/{}/cifar10_{}_weight.npy'.format(classes[index],classes[index])
    magnitude_file_path = './magnitude/cifar10_pgd_training_layer4_class{}_1e-2_1.npy'.format(index)
    frequency_file_path = './Frequency/cifar10_pgd_training_layer4_class{}_1e-2_1.npy'.format(index)
    res1 = np.load(weight_file_path_for_natural_std_training)
    res3 = np.load(magnitude_file_path)
    res4 = np.load(frequency_file_path)

    print(res1)# 正常训练下的class0的weight
    # print(res2)# 对抗训练下的class0的weight
    res3 = res3.reshape(2,512)
    res4 = res4.reshape(2,512)
    res3_attack =res3[0]
    res3_natural = res3[1]
    res4_attack =res4[0]
    res4_natural = res4[1]
    print(res3_natural)
    print(res3_attack)
    print(res4_natural)
    print(res4_attack)
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    saved_path = './layersaved/pgd_training_natural_and_pgd_attack/{}'.format(classes[index])
    if os.path.exists(saved_path) == False:
        os.makedirs(saved_path)

    test = pd.DataFrame(data=res1)
    test.to_csv(saved_path+'/linear(weight)_pgdtrain.csv'.format(classes[index]), encoding='gbk')
    test = pd.DataFrame(data=res3_natural)
    test.to_csv(saved_path+'/l4(magnitude)_pgdtrain_for_natural.csv'.format(classes[index]), encoding='gbk')
    test = pd.DataFrame(data=res3_attack)
    test.to_csv(saved_path+'/l4(magnitude)_pgdtrain_for_pgd.csv'.format(classes[index]), encoding='gbk')
    test = pd.DataFrame(data=res4_natural)
    test.to_csv(saved_path+'/l4(Frequency)_pgdtrain_for_natural.csv'.format(classes[index]), encoding='gbk')
    test = pd.DataFrame(data=res4_attack)
    test.to_csv(saved_path+'/l4(Frequency)_pgdtrain_for_pgd.csv'.format(classes[index]), encoding='gbk')
