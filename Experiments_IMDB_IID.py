#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#================================= Start of importing required packages and libraries =========================================#
from __future__ import print_function
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import torch
from experiment_federated import *
import random
#================================== End of importing required packages and libraries ==========================================#


# In[ ]:


#=============================== Defining global variables ========================#
DATASET_NAME = "IMDB"
MODEL_NAME = "BiLSTM"
DD_TYPE = 'IID'
ALPHA = 1
NUM_PEERS = 20 # "number of peers: K" 
FRAC_PEERS = 1 #'the fraction of peers: C to bel selected in each round'
SEED = 7 #fixed seed
random.seed(SEED)
CRITERION = nn.BCEWithLogitsLoss()
GLOBAL_ROUNDS = 50 #"number of rounds of federated model training"
LOCAL_EPOCHS = 1 #"the number of local epochs: E for each peer"
TEST_BATCH_SIZE = 100
LOCAL_BS = 32 #"local batch size: B for each peer"
LOCAL_LR =  0.001#local learning rate: lr for each peer
LOCAL_MOMENTUM = 0.9 #local momentum for each peer
NUM_CLASSES = 10 # number of classes in an experiment

LABELS_DICT = {'Negative':0,
               'Positive':1}

#select the device to work with cpu or gpu
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
DEVICE = torch.device(DEVICE)
SOURCE_CLASS = 1 # the source class positive
TARGET_CLASS = 0 # the target class negative

CLASS_PER_PEER = 1
SAMPLES_PER_CLASS = 582
RATE_UNBALANCE = 1


# In[ ]:


# Baseline|: FedAvg-no attacks (FL)
RULE = 'fedavg'
ATTACK_TYPE='label_flipping'
MALICIOUS_BEHAVIOR_RATE = 1
for atr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    run_exp(dataset_name = DATASET_NAME, model_name = MODEL_NAME, dd_type = DD_TYPE, num_peers = NUM_PEERS, 
            frac_peers = FRAC_PEERS, seed = SEED, test_batch_size = TEST_BATCH_SIZE,
                criterion = CRITERION, global_rounds = GLOBAL_ROUNDS, local_epochs = LOCAL_EPOCHS, local_bs = LOCAL_BS, 
                 local_lr = LOCAL_LR, local_momentum = LOCAL_MOMENTUM, labels_dict = LABELS_DICT, device = DEVICE,
                attackers_ratio = atr, attack_type=ATTACK_TYPE, 
                 malicious_behavior_rate = MALICIOUS_BEHAVIOR_RATE, rule = RULE,
                source_class = SOURCE_CLASS, target_class = TARGET_CLASS,
               class_per_peer = CLASS_PER_PEER, samples_per_class = SAMPLES_PER_CLASS,
                rate_unbalance = RATE_UNBALANCE, alpha = ALPHA, resume = False)


# In[ ]:


RULE = 'median'
ATTACK_TYPE='label_flipping'
MALICIOUS_BEHAVIOR_RATE = 1
for atr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    run_exp(dataset_name = DATASET_NAME, model_name = MODEL_NAME, dd_type = DD_TYPE, num_peers = NUM_PEERS, 
            frac_peers = FRAC_PEERS, seed = SEED, test_batch_size = TEST_BATCH_SIZE,
                criterion = CRITERION, global_rounds = GLOBAL_ROUNDS, local_epochs = LOCAL_EPOCHS, local_bs = LOCAL_BS, 
                    local_lr = LOCAL_LR, local_momentum = LOCAL_MOMENTUM, labels_dict = LABELS_DICT, device = DEVICE,
                attackers_ratio = atr, attack_type=ATTACK_TYPE, 
                    malicious_behavior_rate = MALICIOUS_BEHAVIOR_RATE, rule = RULE,
                source_class = SOURCE_CLASS, target_class = TARGET_CLASS,
                class_per_peer = CLASS_PER_PEER, samples_per_class = SAMPLES_PER_CLASS,
                rate_unbalance = RATE_UNBALANCE, alpha = ALPHA, resume = False)


# In[ ]:


RULE = 'rmedian'
ATTACK_TYPE='label_flipping'
MALICIOUS_BEHAVIOR_RATE = 1
for atr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    run_exp(dataset_name = DATASET_NAME, model_name = MODEL_NAME, dd_type = DD_TYPE, num_peers = NUM_PEERS, 
            frac_peers = FRAC_PEERS, seed = SEED, test_batch_size = TEST_BATCH_SIZE,
                criterion = CRITERION, global_rounds = GLOBAL_ROUNDS, local_epochs = LOCAL_EPOCHS, local_bs = LOCAL_BS, 
                    local_lr = LOCAL_LR, local_momentum = LOCAL_MOMENTUM, labels_dict = LABELS_DICT, device = DEVICE,
                attackers_ratio = atr, attack_type=ATTACK_TYPE, 
                    malicious_behavior_rate = MALICIOUS_BEHAVIOR_RATE, rule = RULE,
                source_class = SOURCE_CLASS, target_class = TARGET_CLASS,
                class_per_peer = CLASS_PER_PEER, samples_per_class = SAMPLES_PER_CLASS,
                rate_unbalance = RATE_UNBALANCE, alpha = ALPHA, resume = False)


# In[ ]:


RULE = 'tmean'
ATTACK_TYPE='label_flipping'
MALICIOUS_BEHAVIOR_RATE = 1
for atr in [0.0, 0.1, 0.2, 0.3, 0.4]:
    run_exp(dataset_name = DATASET_NAME, model_name = MODEL_NAME, dd_type = DD_TYPE, num_peers = NUM_PEERS, 
            frac_peers = FRAC_PEERS, seed = SEED, test_batch_size = TEST_BATCH_SIZE,
                criterion = CRITERION, global_rounds = GLOBAL_ROUNDS, local_epochs = LOCAL_EPOCHS, local_bs = LOCAL_BS, 
                    local_lr = LOCAL_LR, local_momentum = LOCAL_MOMENTUM, labels_dict = LABELS_DICT, device = DEVICE,
                attackers_ratio = atr, attack_type=ATTACK_TYPE, 
                    malicious_behavior_rate = MALICIOUS_BEHAVIOR_RATE, rule = RULE,
                source_class = SOURCE_CLASS, target_class = TARGET_CLASS,
                class_per_peer = CLASS_PER_PEER, samples_per_class = SAMPLES_PER_CLASS,
                rate_unbalance = RATE_UNBALANCE, alpha = ALPHA, resume = False)


# In[ ]:


RULE = 'mkrum'
ATTACK_TYPE='label_flipping'
MALICIOUS_BEHAVIOR_RATE = 1
for atr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    run_exp(dataset_name = DATASET_NAME, model_name = MODEL_NAME, dd_type = DD_TYPE, num_peers = NUM_PEERS, 
            frac_peers = FRAC_PEERS, seed = SEED, test_batch_size = TEST_BATCH_SIZE,
                criterion = CRITERION, global_rounds = GLOBAL_ROUNDS, local_epochs = LOCAL_EPOCHS, local_bs = LOCAL_BS, 
                    local_lr = LOCAL_LR, local_momentum = LOCAL_MOMENTUM, labels_dict = LABELS_DICT, device = DEVICE,
                attackers_ratio = atr, attack_type=ATTACK_TYPE, 
                    malicious_behavior_rate = MALICIOUS_BEHAVIOR_RATE, rule = RULE,
                source_class = SOURCE_CLASS, target_class = TARGET_CLASS,
                class_per_peer = CLASS_PER_PEER, samples_per_class = SAMPLES_PER_CLASS,
                rate_unbalance = RATE_UNBALANCE, alpha = ALPHA, resume = False)


# In[ ]:


RULE = 'foolsgold'
ATTACK_TYPE='label_flipping'
MALICIOUS_BEHAVIOR_RATE = 1
for atr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    run_exp(dataset_name = DATASET_NAME, model_name = MODEL_NAME, dd_type = DD_TYPE, num_peers = NUM_PEERS, 
            frac_peers = FRAC_PEERS, seed = SEED, test_batch_size = TEST_BATCH_SIZE,
                criterion = CRITERION, global_rounds = GLOBAL_ROUNDS, local_epochs = LOCAL_EPOCHS, local_bs = LOCAL_BS, 
                    local_lr = LOCAL_LR, local_momentum = LOCAL_MOMENTUM, labels_dict = LABELS_DICT, device = DEVICE,
                attackers_ratio = atr, attack_type=ATTACK_TYPE, 
                    malicious_behavior_rate = MALICIOUS_BEHAVIOR_RATE, rule = RULE,
                source_class = SOURCE_CLASS, target_class = TARGET_CLASS,
                class_per_peer = CLASS_PER_PEER, samples_per_class = SAMPLES_PER_CLASS,
                rate_unbalance = RATE_UNBALANCE, alpha = ALPHA, resume = False)


# In[ ]:


RULE = 'lfa_defender'
ATTACK_TYPE='label_flipping'
MALICIOUS_BEHAVIOR_RATE = 1
for atr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    run_exp(dataset_name = DATASET_NAME, model_name = MODEL_NAME, dd_type = DD_TYPE, num_peers = NUM_PEERS, 
            frac_peers = FRAC_PEERS, seed = SEED, test_batch_size = TEST_BATCH_SIZE,
                criterion = CRITERION, global_rounds = GLOBAL_ROUNDS, local_epochs = LOCAL_EPOCHS, local_bs = LOCAL_BS, 
                    local_lr = LOCAL_LR, local_momentum = LOCAL_MOMENTUM, labels_dict = LABELS_DICT, device = DEVICE,
                attackers_ratio = atr, attack_type=ATTACK_TYPE, 
                    malicious_behavior_rate = MALICIOUS_BEHAVIOR_RATE, rule = RULE,
                source_class = SOURCE_CLASS, target_class = TARGET_CLASS,
                class_per_peer = CLASS_PER_PEER, samples_per_class = SAMPLES_PER_CLASS,
                rate_unbalance = RATE_UNBALANCE, alpha = ALPHA, resume = False)


# In[ ]:




