import copy
import enum
import torch
import numpy as np
import math
from scipy import stats
from functools import reduce
import time
import sklearn.metrics.pairwise as smp
import hdbscan
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.cluster import KMeans
from utils import *


eps = np.finfo(float).eps

class LFD():
    def __init__(self, num_classes):
        self.memory = np.zeros([num_classes])
    
    def clusters_dissimilarity(self, clusters, centers):
        n0 = len(clusters[0])
        n1 = len(clusters[1])
        m = n0 + n1 
        cs0 = smp.cosine_similarity(clusters[0])
        cs1 = smp.cosine_similarity(clusters[1])
        mincs0 = np.min(cs0, axis=1)
        mincs1 = np.min(cs1, axis=1)
        ds0 = n0/m * (1 - np.mean(mincs0))
        ds1 = n1/m * (1 - np.mean(mincs1))
        return ds0, ds1

    def aggregate_mild(self, global_model, local_models, ptypes):
        local_weights = [copy.deepcopy(model).state_dict() for model in local_models]
        m = len(local_models)
        for i in range(m):
            local_models[i] = list(local_models[i].parameters())
        global_model = list(global_model.parameters())
        dw = [None for i in range(m)]
        db = [None for i in range(m)]
        for i in range(m):
            dw[i]= global_model[-2].cpu().data.numpy() - \
                local_models[i][-2].cpu().data.numpy() 
            
            #print(dw[i])
            #print("next db")
            
            
            db[i]= global_model[-1].cpu().data.numpy() - \
                local_models[i][-1].cpu().data.numpy()
            
            #print(db[i])
        dw = np.asarray(dw)
        #print(dw)
        #print("next db")
        db = np.asarray(db)
        #print(db)

        "If one class or two classes classification model"
        if len(db[0]) <= 2:
            data = []
            for i in range(m):
                data.append(dw[i].reshape(-1))
        
            kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
            labels = kmeans.labels_

            clusters = {0:[], 1:[]}
            for i, l in enumerate(labels):
                clusters[l].append(data[i])

            good_cl = 0
            cs0, cs1 = self.clusters_dissimilarity(clusters, kmeans.cluster_centers_)
            if cs0 < cs1:
                good_cl = 1

            # print('Cluster 0 weighted variance', cs0)
            # print('Cluster 1 weighted variance', cs1)
            # print('Potential good cluster is:', good_cl)
            scores = np.ones([m])
            for i, l in enumerate(labels):
                # print(ptypes[i], 'Cluster:', l)
                if l != good_cl:
                    scores[i] = 0
                
            global_weights = average_weights(local_weights, scores)
            return global_weights

        "For multiclassification models"
        norms = np.linalg.norm(dw, axis = -1) 
        self.memory = np.sum(norms, axis = 0)
        self.memory +=np.sum(abs(db), axis = 0)
        max_two_freq_classes = self.memory.argsort()[-2:]
        #print('Potential source and target classes:', max_two_freq_classes)
        data = []
        for i in range(m):
            data.append(dw[i][max_two_freq_classes].reshape(-1))

        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
        labels = kmeans.labels_

        clusters = {0:[], 1:[]}
        for i, l in enumerate(labels):
          clusters[l].append(data[i])

        good_cl = 0
        cs0, cs1 = self.clusters_dissimilarity(clusters, kmeans.cluster_centers_)
        if cs0 < cs1:
            good_cl = 1

        # print('Cluster 0 weighted variance', cs0)
        # print('Cluster 1 weighted variance', cs1)
        # print('Potential good cluster is:', good_cl)
        scores = np.ones([m])
        for i, l in enumerate(labels):
            # print(ptypes[i], 'Cluster:', l)
            if l != good_cl:
                scores[i] = 0
            
        global_weights = average_weights(local_weights, scores)
        return global_weights
    def aggregate_extreme(self, global_model, local_models, minimum_cluster_size, peers_types):
        m = len(local_models)
        local_weights = [copy.deepcopy(model).state_dict() for model in local_models]
        global_model = list(global_model.parameters())        
        for i in range(m):
            local_models[i] = list(local_models[i].parameters())
        
        dw = [None for i in range(m)]
        for i in range(m):
            dw[i]= global_model[-2].cpu().data.numpy() - \
                local_models[i][-2].cpu().data.numpy() 
        dw = np.asarray(dw)

        norms = np.linalg.norm(dw, axis = -1) 
        max_two_classes = np.argsort(norms, axis=1)[:, -2:].reshape(-1, 2)
      
        grads = []
        data = []
        for i in range(m):
            grads.append(dw[i])
            data.append(dw[i][max_two_classes[i]].reshape(-1))

        clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=minimum_cluster_size)
        clusterer.fit(data)
        labels = clusterer.labels_
    
        lbls = set(labels)
        # for i, l in enumerate(labels):
        #     print(peers_types[i], l)
        
        clusters = {label:{'grads':[], 'local_weights':[]} for label in lbls}
        for cluster in clusters.keys():
            cluster_idxs = (labels == cluster)
            for i in range(m):
                if cluster_idxs[i]:
                    clusters[cluster]['grads'].append(grads[i])
                    clusters[cluster]['local_weights'].append(local_weights[i]) 
        
        for cluster in clusters.keys():
            clusters[cluster]['gards_centroid'] = np.mean(clusters[cluster]['grads'], axis = 0)
            clusters[cluster]['weights_centroid'] = average_weights(clusters[cluster]['local_weights'], 
                                                    [1 for i in range(len(clusters[cluster]['local_weights']))])
            
            neurons_norms = np.linalg.norm(clusters[cluster]['gards_centroid'], axis = -1) 
            max_norm_neuron = np.argmax(neurons_norms)
            clusters[cluster]['max_norm_neuron'] = max_norm_neuron
            if cluster == -1:
                clusters[cluster]['score'] = 0
            else:
                clusters[cluster]['score'] = 1
        
        keys = list(clusters.keys())
        n = len(keys)
        for i in range(n):
            for j in range(i, n):
                if i != j:
                    if clusters[keys[i]]['max_norm_neuron'] == clusters[keys[j]]['max_norm_neuron']:
                        if len(clusters[keys[j]]['grads']) < len(clusters[keys[i]]['grads']):
                            clusters[keys[j]]['score'] = 0
                            print('Bad cluster', keys[j])
                        else:
                            clusters[keys[i]]['score'] = 0
                            print('Bad cluster', keys[i])

        clusters_weights_centroids = []
        scores = []
        for cluster in clusters.keys():
            clusters_weights_centroids.append(clusters[cluster]['weights_centroid'])
            scores.append(clusters[cluster]['score'])

        global_weights = average_weights(clusters_weights_centroids, scores).reshape(-1, 1)
        return global_weights
        
    def ppfl_cossim(self, global_model, local_grads):
        #local_grads = [copy.deepcopy(model).state_dict() for model in local_models]
        m = len(local_grads)
        grad_len = np.array(local_grads[0][-2].cpu().data.numpy().shape).prod()
        
        #for i in range(m):
        #    local_models[i] = list(local_models[i].parameters())
        global_model = list(global_model.parameters())
        
        global_model = np.reshape(global_model[-2].cpu().data.numpy(), (grad_len))
        #print(global_model)
        v1 = global_model
        norm_v1 = np.linalg.norm(global_model) 
        
        #print("next grad")
        
        grads = np.zeros((m, grad_len))
        
        dw = [None for i in range(m)]
        for i in range(m):
            grads[i] = np.reshape(local_grads[i][-2].cpu().data.numpy(), (grad_len))#local_grads[i].cpu().data.numpy().reshape(-1,1)
            #print(grads[i])
            v2 = grads[i]
            norm_v2 = np.linalg.norm(grads[i]) 
            dw[i] = v1.dot(v2) / (norm_v1 * norm_v2)
            
        
            #dw[i] = smp.cosine_similarity(grads[i],global_model)
            
        '''
        v1 = global_model[-2].cpu().data.numpy().reshape(-1)
        norm_v1 = np.linalg.norm(v1) 
        for i in range(m):
            v2 = local_models[i][-2].cpu().data.numpy().reshape(-1)
            norm_v2 = np.linalg.norm(v2) 
            dw[i] = v1.dot(v2) / (norm_v1 * norm_v2)
        dw = abs(np.asarray(dw))
        '''
        dw1 = np.asarray(dw)
        dw1 = np.add(np.divide(dw1, 2), 0.5)
        
        for index in range(len(dw)):
            dw1[index] = 1 / (1 + np.exp(-50*(dw1[index]-0.5)))
        
        
        #global_weights = average_weights(local_weights, dw)
        return dw1#global_weights


class FoolsGold:
    def __init__(self, num_peers):
        self.memory = None
        self.wv_history = []
        self.num_peers = num_peers
       
    def score_gradients(self, local_grads, selectec_peers):
        m = len(local_grads)
        grad_len = np.array(local_grads[0][-2].cpu().data.numpy().shape).prod()
        if self.memory is None:
            self.memory = np.zeros((self.num_peers, grad_len))

        grads = np.zeros((m, grad_len))
        for i in range(m):
            grads[i] = np.reshape(local_grads[i][-2].cpu().data.numpy(), (grad_len))
        self.memory[selectec_peers]+= grads
        wv = foolsgold(self.memory)  # Use FG
        self.wv_history.append(wv)
        return wv[selectec_peers]   

# Takes in grad
# Compute similarity
# Get weightings
def foolsgold(grads):
    n_clients = grads.shape[0]
    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    return wv


def median_opt(input):
    shape = input.shape
    input = input.sort()[0]
    if shape[-1] % 2 != 0:
        output = input[..., int((shape[-1] - 1) / 2)]
    else:
        output = (input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]) / 2.0
    return output

def Repeated_Median_Shard(w):
    SHARD_SIZE = 100000
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        if total_num < SHARD_SIZE:
            slopes, intercepts = repeated_median(y)
            y = intercepts + slopes * (len(w) - 1) / 2.0
        else:
            y_result = torch.FloatTensor(total_num).to(device)
            assert total_num == y.shape[0]
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y_shard = y[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                slopes_shard, intercepts_shard = repeated_median(y_shard)
                y_shard = intercepts_shard + slopes_shard * (len(w) - 1) / 2.0
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE] = y_shard
            y = y_result
        y = y.reshape(shape)
        w_med[k] = y
    return w_med


def repeated_median(y):
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yyj = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyi = yyj.transpose(-1, -2)
    xx = torch.FloatTensor(range(num_models)).to(y.device)
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.Tensor([float('Inf')] * num_models).to(y.device)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + diag
    slopes, _ = slopes.sort()
    slopes = median_opt(slopes[:, :, :-1])
    slopes = median_opt(slopes)

    # get intercepts (intercept of median)
    yy_median = median_opt(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.Tensor(xx_median).to(y.device)
    intercepts = yy_median - slopes * xx_median

    return slopes, intercepts


# Repeated Median estimator
def Repeated_Median(w):
    cur_time = time.time()
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        slopes, intercepts = repeated_median(y)
        y = intercepts + slopes * (len(w) - 1) / 2.0

        y = y.reshape(shape)
        w_med[k] = y

    print('repeated median aggregation took {}s'.format(time.time() - cur_time))
    return w_med





        
# simple median estimator
def simple_median(w):
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        median_result = median_opt(y)
        assert total_num == len(median_result)

        weight = torch.reshape(median_result, shape)
        w_med[k] = weight
    return w_med

def trimmed_mean(w, trim_ratio):
    if trim_ratio == 0:
        return average_weights(w, [1 for i in range(len(w))])
        
    assert trim_ratio < 0.5, 'trim ratio is {}, but it should be less than 0.5'.format(trim_ratio)
    trim_num = int(trim_ratio * len(w))
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        y_sorted = y.sort()[0]
        result = y_sorted[:, trim_num:-trim_num]
        result = result.mean(dim=-1)
        assert total_num == len(result)

        weight = torch.reshape(result, shape)
        w_med[k] = weight
    return w_med


# Get average weights
def average_weights(w, marks):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    #print(w_avg)
    #w_previous = torch.load('./checkpoints/MNIST_CNNMNIST_IID_ppfl_defender_0.0.t7')
    #print(w_previous)
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * marks[i]
        w_avg[key] = w_avg[key] *(1/sum(marks))
    return w_avg

    

def Krum(updates, f, multi = False):
    n = len(updates)
    updates = [torch.nn.utils.parameters_to_vector(update.parameters()) for update in updates]
    updates_ = torch.empty([n, len(updates[0])])
    for i in range(n):
      updates_[i] = updates[i]
    k = n - f - 2
    # collection distance, distance from points to points，欧氏距离
    cdist = torch.cdist(updates_, updates_, p=2)
    dist, idxs = torch.topk(cdist, k , largest=False)
    dist = dist.sum(1)
    idxs = dist.argsort()
    if multi:
      return idxs[:k]
    else:
      return idxs[0]
##################################################################
