#Experiments_CIFAR10_MILD_NON_IID

#================================= Start of importing required packages and libraries =========================================#
from __future__ import print_function
%matplotlib inline
import numpy as np
import torch
from experiment_federated import *
import random
#================================== End of importing required packages and libraries ==========================================#


#=============================== Defining global variables ========================#
DATASET_NAME = "CIFAR10"
MODEL_NAME = "ResNet18"
DD_TYPE = 'MILD_NON_IID'
ALPHA = 1
NUM_PEERS = 10 # "number of peers: K" 
FRAC_PEERS = 1 #'the fraction of peers: C to bel selected in each round'
SEED = 7 #fixed seed
random.seed(SEED)
CRITERION = nn.CrossEntropyLoss()
GLOBAL_ROUNDS = 200 #"number of rounds of federated model training"
LOCAL_EPOCHS = 3 #"the number of local epochs: E for each peer"
TEST_BATCH_SIZE = 1000
LOCAL_BS = 32 #"local batch size: B for each peer"
LOCAL_LR =  0.01#local learning rate: lr for each peer
LOCAL_MOMENTUM = 0.9 #local momentum for each peer
NUM_CLASSES = 10 # number of classes in an experiment
LABELS_DICT = {'Plane':0,
               'Car':1, 
               'Bird':2, 
               'Cat':3, 
               'Deer':4,
               'Dog':5, 
               'Frog':6, 
               'Horse':7, 
               'Ship':8, 
               'Truck':9}

#select the device to work with cpu or gpu
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
DEVICE = torch.device(DEVICE)
SOURCE_CLASS = 5 # the source class
TARGET_CLASS = 3 # the target class 

CLASS_PER_PEER = 1
SAMPLES_PER_CLASS = 582
RATE_UNBALANCE = 1


# Baseline|: FedAvg-no attacks (FL)
'''
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
'''

RULE = 'ppfl_defender'
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