import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from datasets import *
from aggregation import *

eps = np.finfo(float).eps

def gaussian_attack(update, peer_pseudonym, malicious_behavior_rate = 0, 
    device = 'cpu', attack = False, mean = 0.0, std = 0.5):
    flag = 0
    for key in update.keys():
        r = np.random.random()
        if r <= malicious_behavior_rate:
            # print('Gausiian noise attack launched by ', peer_pseudonym, ' targeting ', key, i+1)
            noise = torch.cuda.FloatTensor(update[key].shape).normal_(mean=mean, std=std)
            flag = 1
            #update[key]+= noise
            update[key] = -update[key]
    return update, flag

def contains_class(dataset, source_class):
    """
    检查数据集中是否包含特定类别的元素。

    参数:
    dataset: 一个数据集，其中每个元素是一个包含两个元素的元组（x, y）。
    source_class: 需要检查的类别。

    返回:
    如果数据集中至少有一个元素的类别与source_class匹配，则返回True，否则返回False。
    """
    # 遍历数据集中的每个元素
    for i in range(len(dataset)):
        # 解包元组，x为数据，y为类别
        x, y = dataset[i]
        # 检查当前元素的类别是否与source_class匹配
        if y == source_class:
            # 如果匹配，返回True
            return True
    # 如果遍历完所有元素都没有匹配的类别，返回False
    return False


# Prepare the dataset for label flipping attack from a target class to another class
def label_filp(data, source_class, target_class):
    """
    创建一个中毒数据集，通过将特定类别的标签翻转为目标类别的标签

    参数:
    data: 原始数据集，可以是一个列表或者支持索引的数据结构
    source_class: 需要进行标签翻转的原始类别
    target_class: 将原始类别标签翻转后的目标类别

    返回:
    poisoned_data: 中毒数据集对象，用于进一步的数据处理或训练模型
    """
    # 创建一个中毒数据集实例，传入原始数据、源类别和目标类别参数
    poisoned_data = PoisonedDataset(data, source_class, target_class)

    # 返回中毒数据集实例
    return poisoned_data


#Plot the PCA of updates with their peers types. Types are: Honest peer or attacker
def plot_updates_components(updates, peers_types, epoch):
    flattened_updates = flatten_updates(updates)
    flattened_updates = StandardScaler().fit_transform(flattened_updates)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(flattened_updates)
    principalDf = pd.DataFrame(data = principalComponents,
                                columns = ['c1', 'c2'])
    peers_typesDf = pd.DataFrame(data = peers_types,
                                columns = ['target'])
    finalDf = pd.concat([principalDf, peers_typesDf['target']], axis = 1)
    fig = plt.figure(figsize = (7,7))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Component 1', fontsize = 10)
    ax.set_ylabel('Component 2', fontsize = 10)
    ax.set_title('2 component PCA', fontsize = 15)
    targets = ['Good update', 'Bad update']
    colors = ['white', 'black']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'c1'], 
                    finalDf.loc[indicesToKeep, 'c2'], 
                    c = color, 
                    edgecolors='gray',
                    s = 80)
    ax.legend(targets)
    plt.savefig('pca\epoch{}.png'.format(epoch), dpi = 600)
    # plt.show()

def plot_layer_components(updates, peers_types, epoch, layer = 'linear.weight'):
   
    res = {'updates':updates, 'peers_types':peers_types}
    torch.save(res, 'results/epoch{}.t7'.format(epoch))

    layers = ['linear.weight', 'linear.bias']
    flattened_updates = flatten_updates(updates, layers = layers)
    flattened_updates = StandardScaler().fit_transform(flattened_updates)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(flattened_updates)
    principalDf = pd.DataFrame(data = principalComponents,
                                columns = ['c1', 'c2'])
    peers_typesDf = pd.DataFrame(data = peers_types,
                                columns = ['target'])
    finalDf = pd.concat([principalDf, peers_typesDf['target']], axis = 1)
    fig = plt.figure(figsize = (7,7))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Component 1', fontsize = 10)
    ax.set_ylabel('Component 2', fontsize = 10)
    ax.set_title('2 component PCA', fontsize = 15)
    targets = ['Good update', 'Bad update']
    colors = ['white', 'black']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'c1'], 
                    finalDf.loc[indicesToKeep, 'c2'], 
                    c = color, 
                    edgecolors='gray',
                    s = 80)
    ax.legend(targets)
    plt.savefig('pca\epoch{}_layer_{}.png'.format(epoch, layer), dpi = 600)
    plt.show()



