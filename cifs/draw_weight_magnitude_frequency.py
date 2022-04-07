import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
testdata = ('natural', 'fgsm', 'pgd_10', 'cw')
training_model = ('std', 'adv')

for index in range(10):


    plt.figure()
    path0 = './layersaved/std_training_natural_and_fgsm_attack/'+classes[index]+'/l4(magnitude)_stdtrain_for_natural.csv'.format(index)
    path1 = './layersaved/std_training_natural_and_fgsm_attack/'+classes[index]+'/l4(magnitude)_stdtrain_for_fgsm.csv'.format(index)
    path2 = './layersaved/std_training_natural_and_fgsm_attack/'+classes[index]+'/l4(Frequency)_stdtrain_for_natural.csv'.format(index)
    path3 = './layersaved/std_training_natural_and_fgsm_attack/'+classes[index]+'/l4(Frequency)_stdtrain_for_fgsm.csv'.format(index)
    path4 = './layersaved/std_training_natural_and_fgsm_attack/'+classes[index]+'/linear(weight)_stdtrain.csv'.format(index)

    path5 = './layersaved/std_training_natural_and_pgd_attack/'+classes[index]+'/l4(magnitude)_stdtrain_for_natural.csv'.format(index)
    path6 = './layersaved/std_training_natural_and_pgd_attack/'+classes[index]+'/l4(magnitude)_stdtrain_for_pgd.csv'.format(index)
    path7 = './layersaved/std_training_natural_and_pgd_attack/'+classes[index]+'/l4(Frequency)_stdtrain_for_natural.csv'.format(index)
    path8 = './layersaved/std_training_natural_and_pgd_attack/'+classes[index]+'/l4(Frequency)_stdtrain_for_pgd.csv'.format(index)
    path9 = './layersaved/std_training_natural_and_pgd_attack/'+classes[index]+'/linear(weight)_stdtrain.csv'.format(index)


    data0 = np.array((pd.read_csv(path0,usecols=[1],encoding='gbk'))["0"])
    data1 = np.array((pd.read_csv(path1,usecols=[1],encoding='gbk'))["0"])
    data2 = np.array((pd.read_csv(path2,usecols=[1],encoding='gbk'))["0"])
    data3 = np.array((pd.read_csv(path3,usecols=[1],encoding='gbk'))["0"])
    data4 = np.array((pd.read_csv(path4,usecols=[1],encoding='gbk'))["0"])

    data4_index = np.argsort(data4)
    data4_sort = data4[data4_index]

    data0_index = np.argsort(data4)
    data0_sort = data0[data0_index]
    data1_sort = data1[data0_index]

    data2_index = np.argsort(data4)
    data2_sort = data2[data2_index]
    data3_sort = data3[data2_index]

    data5 = np.array((pd.read_csv(path5,usecols=[1],encoding='gbk'))["0"])
    data6 = np.array((pd.read_csv(path6,usecols=[1],encoding='gbk'))["0"])
    data7 = np.array((pd.read_csv(path7,usecols=[1],encoding='gbk'))["0"])
    data8 = np.array((pd.read_csv(path8,usecols=[1],encoding='gbk'))["0"])
    data9 = np.array((pd.read_csv(path9,usecols=[1],encoding='gbk'))["0"])

    data9_index = np.argsort(data9)
    data9_sort = data9[data9_index]

    data5_index = np.argsort(data9)
    data5_sort = data5[data5_index]
    data6_sort = data6[data5_index]

    data7_index = np.argsort(data9)
    data7_sort = data7[data7_index]
    data8_sort = data8[data7_index]

    xais0 = range(len(data4))

    ###########################################################################
    fig0=plt.subplot(2,3,1)
    fig0.set_ylim(-1, 1)
    fig0.set_xlim(0.0, 512.0)
    m0=data0_sort[::-1]
    m1=data1_sort[::-1]
    plt.plot(m0,color='blue',alpha=0.7,label='{}_std_training_natural examples'.format(classes[index]))
    plt.plot(m1,color='red',alpha=0.4,label='{}_std_training_fgsm examples'.format(classes[index]))
    fig0.set_xlabel("channel",fontdict={'family': 'Times Roman', 'size': 12})
    fig0.set_ylabel("magnitude",fontdict={'family': 'Times Roman', 'size': 12})
    handles, labels = fig0.get_legend_handles_labels()
    fig0.legend(handles[::-1], labels[::-1], loc=0, ncol=1, prop={'family': 'Times Roman', 'size': 12})
    fig0.tick_params(axis='both', which='major', labelsize=12)
    fig0.tick_params(axis='both', which='minor', labelsize=12)

    plt.grid(linestyle='--')



    ###########################################################################
    fig1=plt.subplot(2,3,2)
    fig1.set_ylim(-1, 1)
    fig1.set_xlim(0.0, 512.0)
    W0=data4_sort[::-1]
    plt.plot(W0,color='blue',alpha=0.7,label='{}_weigth'.format(classes[index]))
    plt.grid(linestyle='--')
    fig1.set_xlabel("channel",fontdict={'family': 'Times Roman', 'size': 12})
    fig1.set_ylabel("weight",fontdict={'family': 'Times Roman', 'size': 12})
    handles, labels = fig1.get_legend_handles_labels()
    fig1.legend(handles[::-1], labels[::-1], loc=0, ncol=1, prop={'family': 'Times Roman', 'size': 12})
    fig1.tick_params(axis='both', which='major', labelsize=12)
    fig1.tick_params(axis='both', which='minor', labelsize=12)



    ###########################################################################
    fig2=plt.subplot(2,3,3)
    fig2.set_ylim(-1000, 1000)
    fig2.set_xlim(0.0, 512.0)
    F0=data2_sort[::-1]
    F1=data3_sort[::-1]
    plt.plot(F0,color='blue',alpha=0.7,label='{}_std_training_natural examples'.format(classes[index]))
    plt.plot(F1,color='red',alpha=0.4,label='{}_std_training_fgsm examples'.format(classes[index]))
    plt.grid(linestyle='--')
    fig2.set_xlabel("channel",fontdict={'family': 'Times Roman', 'size': 12})
    fig2.set_ylabel("Frequency",fontdict={'family': 'Times Roman', 'size': 12})
    handles, labels = fig2.get_legend_handles_labels()
    fig2.legend(handles[::-1], labels[::-1], loc=0, ncol=1, prop={'family': 'Times Roman', 'size': 12})
    fig2.tick_params(axis='both', which='major', labelsize=12)
    fig2.tick_params(axis='both', which='minor', labelsize=12)

    ###########################################################################
    fig3=plt.subplot(2,3,4)
    fig3.set_ylim(-1, 1)
    fig3.set_xlim(0.0, 512.0)
    m00=data5_sort[::-1]
    m11=data6_sort[::-1]
    plt.plot(m00,color='blue',alpha=0.7,label='{}_std_training_natural examples'.format(classes[index]))
    plt.plot(m11,color='red',alpha=0.4,label='{}_std_training_pgd examples'.format(classes[index]))
    fig3.set_xlabel("channel",fontdict={'family': 'Times Roman', 'size': 12})
    fig3.set_ylabel("magnitude",fontdict={'family': 'Times Roman', 'size': 12})
    handles, labels = fig3.get_legend_handles_labels()
    fig3.legend(handles[::-1], labels[::-1], loc=0, ncol=1, prop={'family': 'Times Roman', 'size': 12})
    fig3.tick_params(axis='both', which='major', labelsize=12)
    fig3.tick_params(axis='both', which='minor', labelsize=12)

    plt.grid(linestyle='--')



    ###########################################################################
    fig4=plt.subplot(2,3,5)
    fig4.set_ylim(-1, 1)
    fig4.set_xlim(0.0, 512.0)
    W00=data9_sort[::-1]
    plt.plot(W00,color='blue',alpha=0.7,label='{}_weigth'.format(classes[index]))
    plt.grid(linestyle='--')
    fig4.set_xlabel("channel",fontdict={'family': 'Times Roman', 'size': 12})
    fig4.set_ylabel("weight",fontdict={'family': 'Times Roman', 'size': 12})
    handles, labels = fig4.get_legend_handles_labels()
    fig4.legend(handles[::-1], labels[::-1], loc=0, ncol=1, prop={'family': 'Times Roman', 'size': 12})
    fig4.tick_params(axis='both', which='major', labelsize=12)
    fig4.tick_params(axis='both', which='minor', labelsize=12)



    ###########################################################################
    fig5=plt.subplot(2,3,6)
    fig5.set_ylim(-1000, 1000)
    fig5.set_xlim(0.0, 512.0)
    F00=data7_sort[::-1]
    F11=data8_sort[::-1]
    plt.plot(F00,color='blue',alpha=0.7,label='{}_std_training_natural examples'.format(classes[index]))
    plt.plot(F11,color='red',alpha=0.4,label='{}_std_training_pgd examples'.format(classes[index]))
    plt.grid(linestyle='--')
    fig5.set_xlabel("channel",fontdict={'family': 'Times Roman', 'size': 12})
    fig5.set_ylabel("Frequency",fontdict={'family': 'Times Roman', 'size': 12})
    handles, labels = fig5.get_legend_handles_labels()
    fig5.legend(handles[::-1], labels[::-1], loc=0, ncol=1, prop={'family': 'Times Roman', 'size': 12})
    fig5.tick_params(axis='both', which='major', labelsize=12)
    fig5.tick_params(axis='both', which='minor', labelsize=12)
    plt.show()









