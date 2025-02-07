'''Some helper functions
'''
import random
from random import shuffle

random.seed(7)
import numpy as np
from torchvision import datasets, transforms
import codecs
import tensorflow as tf
import pandas as pd
from datasets import *


def distribute_dataset(dataset_name, num_peers, num_classes, dd_type='IID', classes_per_peer=1, samples_per_class=582,
                       alpha=1):
    print("--> Loading of {} dataset".format(dataset_name))
    tokenizer = None
    if dataset_name == 'MNIST':
        trainset, testset = get_mnist()
    elif dataset_name == 'CIFAR10':
        trainset, testset = get_cifar10()
    elif dataset_name == 'IMDB':
        trainset, testset, tokenizer = get_imdb(num_peers=num_peers)
    if dd_type == 'IID':
        peers_data_dict = sample_dirichlet(trainset, num_peers, alpha=1000000)
    elif dd_type == 'MILD_NON_IID':
        peers_data_dict = sample_dirichlet(trainset, num_peers, alpha=alpha)
    elif dd_type == 'EXTREME_NON_IID':
        peers_data_dict = sample_extreme(trainset, num_peers, num_classes, classes_per_peer, samples_per_class)

    print("--> Dataset has been loaded!")
    return trainset, testset, peers_data_dict, tokenizer


# Get the original MNIST data set
def get_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = datasets.MNIST('./data', train=True, download=True,
                              transform=transform)
    testset = datasets.MNIST('./data', train=False, download=True,
                             transform=transform)
    return trainset, testset


# Get the original CIFAR10 data set
def get_cifar10():
    data_dir = 'data/cifar/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = datasets.CIFAR10(data_dir, train=True, download=True,
                                transform=apply_transform)

    testset = datasets.CIFAR10(data_dir, train=False, download=True,
                               transform=apply_transform)
    return trainset, testset


# Get the IMDB data set
def get_imdb(num_peers=10):
    """
    准备并返回IMDB数据集用于训练和验证。

    该函数从CSV文件读取IMDB数据集，预处理文本数据（包括分词和填充序列），并将数据集分为训练集和验证集。
    同时将情感标签转换为数值形式。

    参数:
    - num_peers: int, 对手的数量，默认为10。 (此参数在函数中未使用，如果不需要可以移除。)

    返回值:
    - trainset: 训练数据集对象。
    - testset: 验证数据集对象。
    - tokenizer: 分词器对象。
    """

    MAX_LEN = 128
    # Read data
    df = pd.read_csv('data/imdb.csv')
    # Convert sentiment columns to numerical values # 将情感列转换为数值
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == 'positive' else 0)
    # Tokenization
    # use tf.keras for tokenization,
    # 使用tf.keras进行分词
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist())

    train_df = df.iloc[:40000].reset_index(drop=True)
    valid_df = df.iloc[40000:].reset_index(drop=True)

    # STEP 3: pad sequence  填充序列
    xtrain = tokenizer.texts_to_sequences(train_df.review.values)
    xtest = tokenizer.texts_to_sequences(valid_df.review.values)

    # zero padding  零填充
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=MAX_LEN)
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=MAX_LEN)

    # STEP 4: initialize dataset class for training 初始化训练数据集类
    trainset = IMDBDataset(reviews=xtrain, targets=train_df.sentiment.values)

    # initialize dataset class for validation   初始化验证数据集类
    testset = IMDBDataset(reviews=xtest, targets=valid_df.sentiment.values)

    return trainset, testset, tokenizer


def sample_dirichlet(dataset, num_users, alpha=1):
    """
    使用Dirichlet分布对数据集进行采样以分配给多个用户。

    参数:
    dataset: 数据集，通常是一个包含数据样本和标签的列表。
    num_users: 用户数量，决定数据集分割的数量。
    alpha: Dirichlet分布的参数，影响数据分配的平衡性。

    返回:
    peers_data_dict: 包含每个用户数据索引和标签的字典。
    """
    # 初始化一个字典，用于存储每个类别的索引
    classes = {}
    # 遍历数据集，将每个样本的索引分配到其对应的标签类别中
    for idx, x in enumerate(dataset):
        _, label = x
        # 如果标签是Tensor，将其转换为item
        if type(label) == torch.Tensor:
            label = label.item()
        # 如果标签已经存在于classes字典中，追加索引
        if label in classes:
            classes[label].append(idx)
        # 否则，创建一个新的列表并追加索引
        else:
            classes[label] = [idx]
    # 计算类别总数
    num_classes = len(classes.keys())

    # 初始化一个字典，用于存储每个用户的数据索引和标签
    peers_data_dict = {i: {'data': np.array([]), 'labels': set()} for i in range(num_users)}

    # 遍历每个类别，按照Dirichlet分布将数据分配给用户
    for n in range(num_classes):
        # 将类别的索引列表打乱
        random.shuffle(classes[n])
        # 计算当前类别的大小
        class_size = len(classes[n])
        # 按照Dirichlet分布采样，决定每个用户应该获得的数据量比例
        sampled_probabilities = class_size * np.random.dirichlet(np.array(num_users * [alpha]))
        # 遍历每个用户，按照采样比例分配数据
        for user in range(num_users):
            # 计算当前用户应该获得的数据量
            num_imgs = int(round(sampled_probabilities[user]))
            # 从当前类别中采样对应数量的数据索引
            sampled_list = classes[n][:min(len(classes[n]), num_imgs)]
            # 将采样的数据索引添加到用户的数据集中
            peers_data_dict[user]['data'] = np.concatenate((peers_data_dict[user]['data'], np.array(sampled_list)),
                                                           axis=0)
            # 如果采样的数据量大于0，将当前类别的标签添加到用户的标签集中
            if num_imgs > 0:
                peers_data_dict[user]['labels'].add(n)

            # 从当前类别的索引列表中移除已经采样的数据
            classes[n] = classes[n][min(len(classes[n]), num_imgs):]

    # 返回每个用户的数据索引和标签字典
    return peers_data_dict



def sample_extreme(dataset, num_users, num_classes, classes_per_peer, samples_per_class):
    """
    在一个数据集中，为每个用户分配极端样本，即每个用户只分配到特定类别的样本。

    参数:
    dataset: 数据集对象，包含数据和标签。
    num_users: 用户数量，决定数据将被分割成多少部分。
    num_classes: 数据集中的类别总数。
    classes_per_peer: 每个用户分配到的类别数量。
    samples_per_class: 每个类别中每个用户分配到的样本数量。

    返回:
    peers_data_dict: 一个字典，包含每个用户的数据和标签。
    """
    # 获取数据集的大小
    n = len(dataset)
    # 初始化每个用户的数据和标签字典
    peers_data_dict = {i: {'data': np.array([]), 'labels': []} for i in range(num_users)}
    # 创建一个索引数组，用于后续的数据分配
    idxs = np.arange(n)
    # 获取数据集的标签
    labels = np.array(dataset.targets)

    # 将索引和标签堆叠在一起，以便按标签排序
    idxs_labels = np.vstack((idxs, labels))
    # 按标签排序，以便后续按类别分配数据
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    # 初始化每个标签的索引列表
    label_indices = {l: [] for l in range(num_classes)}
    for l in label_indices:
        # 找到每个标签的索引，并将其添加到对应的列表中
        label_idxs = np.where(labels == l)
        label_indices[l] = list(idxs[label_idxs])

    # 初始化标签列表
    labels = [i for i in range(num_classes)]

    for i in range(num_users):
        # 为每个用户随机选择分配的类别
        user_labels = np.random.choice(labels, classes_per_peer, replace=False)
        for l in user_labels:
            # 将选中的类别添加到用户的标签列表中
            peers_data_dict[i]['labels'].append(l)
            # 获取该类别中分配给用户的样本索引
            lab_idxs = label_indices[l][:samples_per_class]
            # 更新该类别的索引列表，移除已分配的样本
            label_indices[l] = list(set(label_indices[l]) - set(lab_idxs))
            # 如果该类别剩余样本不足以分配给下一个用户，则从类别列表中移除
            if len(label_indices[l]) < samples_per_class:
                labels = list(set(labels) - set([l]))
            # 将选中的样本添加到用户的数据列表中
            peers_data_dict[i]['data'] = np.concatenate(
                (peers_data_dict[i]['data'], lab_idxs), axis=0)

    # 返回每个用户的数据和标签字典
    return peers_data_dict

