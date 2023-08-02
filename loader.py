import math
import random
from collections import OrderedDict
import numpy as np
import sys
from utils import get_logger
import scipy.io as scio
from sklearn.model_selection import KFold
import pandas as pd

sys.path.append(".")


def config_model():
    config = OrderedDict()
    return config


def get_disease_sim_Matrix(disease_similarity, disease_disease_topk):
    disease_disease_Matrix = np.zeros((disease_similarity.shape[0], disease_similarity.shape[0]), np.float32)
    disease_sim_Matrix = disease_similarity
    for disease_num0 in range(disease_similarity.shape[0]):
        disease_sim = {}
        for disease_num1 in range(disease_similarity.shape[0]):
            if disease_num0 == disease_num1:
                continue
            disease_sim[disease_num1] = disease_sim_Matrix[disease_num0][disease_num1]
        sorted_disease_list = sorted(disease_sim.items(), key=lambda d: d[1], reverse=True)
        for i in range(min(disease_disease_topk, len(sorted_disease_list))):
            disease_num1 = sorted_disease_list[i][0]
            disease_disease_Matrix[disease_num0][disease_num1] = 1

    return disease_disease_Matrix


def get_drug_sim_Matrix(drug_similarity, drug_drug_topk):
    drug_drug_Matrix = np.zeros((drug_similarity.shape[0], drug_similarity.shape[0]), np.float32)
    drug_sim_Matrix = drug_similarity
    for drug_num0 in range(drug_similarity.shape[0]):
        drug_sim = {}
        for drug_num1 in range(drug_similarity.shape[0]):
            if drug_num0 == drug_num1:
                continue
            drug_sim[drug_num1] = drug_sim_Matrix[drug_num0][drug_num1]
        sorted_drug_list = sorted(drug_sim.items(), key=lambda d: d[1], reverse=True)
        for i in range(min(drug_drug_topk, len(sorted_drug_list))):
            drug_num1 = sorted_drug_list[i][0]
            drug_drug_Matrix[drug_num0][drug_num1] = 1

    return drug_drug_Matrix


def load_mat(filepath):
    mat = scio.loadmat(filepath)
    drug_sim = mat["drug"].astype(np.float)
    disease_sim = mat["disease"].astype(np.float)
    drug_name = mat["Wrname"].reshape(-1)
    drug_num = len(drug_name)
    disease_name = mat["Wdname"].reshape(-1)
    disease_num = len(disease_name)
    interactions = mat["didr"]
    return drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions

def load_lrssl(filepath, reduce=True):
    """ C drug:658, disease:409 association:2520 (False 2353)
        PREDICT drug:593, disease:313 association:1933 (Fdataset)
        LRSSL drug: 763, disease:681, association:3051
    """
    drug_chemical = pd.read_csv(filepath + "lrssl_simmat_dc_chemical.txt", sep="\t", index_col=0)
    drug_dataset = pd.read_csv(filepath + "lrssl_simmat_dc_dataset.txt", sep="\t", index_col=0)
    drug_go = pd.read_csv(filepath + "lrssl_simmat_dc_go.txt", sep="\t", index_col=0)
    disease_sim = pd.read_csv(filepath + "lrssl_simmat_dg.txt", sep="\t", index_col=0)
    if reduce:
        drug_sim = (drug_chemical+drug_dataset+drug_go)/3
    else:
        drug_sim = drug_chemical
    drug_disease = pd.read_csv(filepath + "lrssl_admat_dgc.txt", sep="\t", index_col=0).T
    drug_disease = drug_disease.T
    rr = drug_sim.to_numpy(dtype=np.float32)
    rd = drug_disease.to_numpy(dtype=np.float32)
    dd = disease_sim.to_numpy(dtype=np.float32)
    rname = drug_sim.columns.to_numpy()
    dname = disease_sim.columns.to_numpy()
    drug_sim = rr.astype(np.float)
    disease_sim = dd.astype(np.float)
    drug_name = rname.reshape(-1)
    drug_num = len(drug_name)
    disease_name = dname.reshape(-1)
    disease_num = len(disease_name)
    interactions = rd.T
    return drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions

def load_Ldataset(filepath):
    """drug:598, disease:269 association:18416
    """
    disease_sim = pd.read_csv(filepath + "dis_sim.csv", header=None).to_numpy(dtype=np.float32)
    interactions = pd.read_csv(filepath + "drug_dis.csv", header=None).to_numpy(dtype=np.float32)
    drug_sim = pd.read_csv(filepath + "drug_sim.csv", header=None).to_numpy(dtype=np.float32)
    disease_name = np.arange(disease_sim.shape[0])
    drug_name = np.arange(drug_sim.shape[0])
    disease_num = disease_sim.shape[0]
    drug_num = drug_sim.shape[1]
    return drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions.T

def get_train_adj(adj, train_mask, truth_label):
    for i in range(train_mask.shape[0]):
        for j in range(train_mask.shape[1]):
            if train_mask[i][j]:
                adj[i][j] = truth_label[i][j]
    return adj

def data_reader(logger, config, dataset="Fdataset", disease_disease_topk=7, drug_drug_topk=7):

    path = "./data/" + dataset + "/"
    assert dataset in ["Cdataset", "Fdataset", "Ldataset", "LRSSL"]
    if dataset in ['Fdataset', 'Cdataset']:
        drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions = load_mat(
                                                                                                path + dataset + ".mat")
    elif dataset == "LRSSL":
        drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions = load_lrssl(path)
    else:
        drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions = load_Ldataset(path)

    kfold = KFold(n_splits=config['n_splits'], shuffle=True, random_state=config['seed'])
    pos_row, pos_col = np.nonzero(interactions)
    neg_row, neg_col = np.nonzero(1 - interactions)
    assert len(pos_row) + len(neg_row) == np.prod(interactions.shape)
    train_data, test_data = [], []
    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row),
                                                                            kfold.split(neg_row)):
        train_mask = np.zeros_like(interactions, dtype="bool")
        test_mask = np.zeros_like(interactions, dtype="bool")
        train_pos_edge = np.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])
        train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
        test_pos_edge = np.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
        test_neg_edge = np.stack([neg_row[test_neg_idx], neg_col[test_neg_idx]])
        train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
        test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)
        train_mask[train_edge[0], train_edge[1]] = True
        test_mask[test_edge[0], test_edge[1]] = True
        train_data.append(train_mask)
        test_data.append(test_mask)
    logger.info("drug count :{}".format(drug_num))
    logger.info("disease count:{}".format(disease_num))
    config['disease_size'] = disease_num
    config['drug_size'] = drug_num

    disease_disease_sim_Matrix = get_disease_sim_Matrix(disease_sim, disease_disease_topk)
    drug_drug_sim_Matrix = get_drug_sim_Matrix(drug_sim, drug_drug_topk)
    truth_label = interactions

    return disease_disease_sim_Matrix, drug_drug_sim_Matrix, truth_label, train_data, test_data


def case_data_reader(logger, config, dataset="Fdataset", disease_disease_topk=7, drug_drug_topk=7):

    path = "./data/" + dataset + "/"
    assert dataset in ["Cdataset", "Fdataset", "Ldataset", "LRSSL"]
    if dataset in ['Fdataset', 'Cdataset']:
        drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions = load_mat(
                                                                                                path + dataset + ".mat")
    elif dataset == "LRSSL":
        drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions = load_lrssl(path)
    else:
        drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions = load_Ldataset(path)

    train_data, test_data = [], []
    i = config["specific_id"]
    train, test = {}, {}
    train_disease, train_drug, train_label, test_disease, test_drug, test_label = [], [], [], [], [], []
    train_matrix = interactions.copy()
    for j in range(train_matrix.shape[1]):
        train_matrix[i][j] = 0
    for m in range(train_matrix.shape[0]):
        for n in range(train_matrix.shape[1]):
            train_disease.append(m)
            train_drug.append(n)
            train_label.append(train_matrix[m][n])
    for t in range(train_matrix.shape[1]):
        test_disease.append(i)
        test_drug.append(t)
        test_label.append(interactions[i][t])
    train["disease"] = train_disease
    train["drug"] = train_drug
    train["label"] = train_label
    test["disease"] = test_disease
    test["drug"] = test_drug
    test["label"] = test_label
    train_data.append(train)
    test_data.append(test)
    logger.info("drug count :{}".format(drug_num))
    logger.info("disease count:{}".format(disease_num))
    config['disease_size'] = disease_num
    config['drug_size'] = drug_num
    disease_disease_sim_Matrix = get_disease_sim_Matrix(disease_sim, disease_disease_topk)
    drug_drug_sim_Matrix = get_drug_sim_Matrix(drug_sim, drug_drug_topk)

    return disease_disease_sim_Matrix, drug_drug_sim_Matrix, train_matrix, train_data, test_data


def analysis_data_reader(config, logger, specific_id, dataset="Fdataset", disease_disease_topk=7, drug_drug_topk=7):

    path = "./data/" + dataset + "/"
    assert dataset in ["Cdataset", "Fdataset", "Ldataset", "LRSSL"]
    if dataset in ['Fdataset', 'Cdataset']:
        drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions = load_mat(
                                                                                                path + dataset + ".mat")
    elif dataset == "LRSSL":
        drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions = load_lrssl(path)
    else:
        drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions = load_Ldataset(path)

    train_data, test_data = [], []
    i = specific_id
    train, test = {}, {}
    train_disease, train_drug, train_label, test_disease, test_drug, test_label = [], [], [], [], [], []
    # train_matrix = interactions.copy()
    train_matrix = np.copy(interactions.T)
    for j in range(train_matrix.shape[1]):
        train_matrix[i][j] = 0
    for m in range(train_matrix.shape[0]):
        for n in range(train_matrix.shape[1]):
            train_disease.append(n)
            train_drug.append(m)
            train_label.append(train_matrix[m][n])
    for t in range(train_matrix.shape[1]):
        test_disease.append(t)
        test_drug.append(i)
        test_label.append(interactions[t][i])
    train["disease"] = train_disease
    train["drug"] = train_drug
    train["label"] = train_label
    test["disease"] = test_disease
    test["drug"] = test_drug
    test["label"] = test_label
    train_data.append(train)
    test_data.append(test)
    logger.info("drug count :{}".format(drug_num))
    logger.info("disease count:{}".format(disease_num))
    config['disease_size'] = disease_num
    config['drug_size'] = drug_num
    disease_disease_sim_Matrix = get_disease_sim_Matrix(disease_sim, disease_disease_topk)
    drug_drug_sim_Matrix = get_drug_sim_Matrix(drug_sim, drug_drug_topk)

    return disease_disease_sim_Matrix, drug_drug_sim_Matrix, train_matrix.T, train_data, test_data


class BatchManager(object):

    def __init__(self, data, batch_size, type):
        disease_input, drug_input, labels = [], [], []
        disease_sim_Matrix, drug_sim_Matrix, mask, label = data
        disease_drug_Adj = np.zeros((disease_sim_Matrix.shape[0], drug_sim_Matrix.shape[0]))
        disease_drug_Adj = get_train_adj(disease_drug_Adj, mask, label)
        if type == "train":
            disease_disease_sim_Matrix, drug_drug_sim_Matrix, train_mask, truth_label = data
            for i in range(len(train_mask)):
                for j in range(len(train_mask[0])):
                    if train_mask[i][j]:
                        disease_input.append(i)
                        drug_input.append(j)
                        labels.append(disease_drug_Adj[i][j])
                    else:
                        disease_input.append(i)
                        drug_input.append(j)
                        labels.append(0)
            num_batch = int(math.ceil(len(disease_input) / batch_size))
            self.batch_data = list()

            for i in range(num_batch):
                input_disease = disease_input[i * batch_size: (i + 1) * batch_size]
                input_drug = drug_input[i * batch_size: (i + 1) * batch_size]
                label = labels[i * batch_size: (i + 1) * batch_size]
                self.batch_data.append(
                    [disease_drug_Adj, disease_disease_sim_Matrix, drug_drug_sim_Matrix, input_disease, input_drug,
                     label])

        elif type == "valid" or type == "test":
            disease_disease_sim_Matrix, drug_drug_sim_Matrix, test_mask, truth_label = data
            for i in range(len(test_mask)):
                for j in range(len(test_mask[0])):
                    if test_mask[i][j]:
                        disease_input.append(i)
                        drug_input.append(j)
                        labels.append(truth_label[i][j])
            num_batch = int(math.ceil(len(disease_input) / batch_size))
            self.batch_data = list()
            for i in range(num_batch):
                input_disease = disease_input[i * batch_size: (i + 1) * batch_size]
                input_drug = drug_input[i * batch_size: (i + 1) * batch_size]
                label = labels[i * batch_size: (i + 1) * batch_size]
                self.batch_data.append(
                    [disease_drug_Adj, disease_disease_sim_Matrix, drug_drug_sim_Matrix, input_disease, input_drug,
                     label])

        self.len_data = len(self.batch_data)

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


class case_BatchManager(object):

    def __init__(self, data, batch_size, type):
        disease_sim_Matrix, drug_sim_Matrix, mask, train_matrix = data
        disease_drug_Adj = train_matrix
        if type == "train":
            disease_disease_sim_Matrix, drug_drug_sim_Matrix, train_mask, truth_label = data
            disease_input = train_mask["disease"]
            drug_input = train_mask["drug"]
            labels = train_mask["label"]
            num_batch = int(math.ceil(len(disease_input) / batch_size))
            self.batch_data = list()

            for i in range(num_batch):
                input_disease = disease_input[i * batch_size: (i + 1) * batch_size]
                input_drug = drug_input[i * batch_size: (i + 1) * batch_size]
                label = labels[i * batch_size: (i + 1) * batch_size]
                self.batch_data.append(
                    [disease_drug_Adj, disease_disease_sim_Matrix, drug_drug_sim_Matrix, input_disease, input_drug,
                     label])

        elif type == "valid" or type == "test":
            disease_disease_sim_Matrix, drug_drug_sim_Matrix, test_mask, truth_label = data
            disease_input = test_mask["disease"]
            drug_input = test_mask["drug"]
            labels = test_mask["label"]
            num_batch = int(math.ceil(len(disease_input) / batch_size))
            self.batch_data = list()
            for i in range(num_batch):
                input_disease = disease_input[i * batch_size: (i + 1) * batch_size]
                input_drug = drug_input[i * batch_size: (i + 1) * batch_size]
                label = labels[i * batch_size: (i + 1) * batch_size]
                self.batch_data.append(
                    [disease_drug_Adj, disease_disease_sim_Matrix, drug_drug_sim_Matrix, input_disease, input_drug,
                     label])

        self.len_data = len(self.batch_data)

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


class analysis_BatchManager(object):

    def __init__(self, data, batch_size, type):
        disease_sim_Matrix, drug_sim_Matrix, mask, train_matrix = data
        # disease_drug_Adj = np.zeros((disease_sim_Matrix.shape[0], drug_sim_Matrix.shape[0]))
        disease_drug_Adj = train_matrix
        if type == "train":
            disease_disease_sim_Matrix, drug_drug_sim_Matrix, train_mask, truth_label = data
            disease_input = train_mask["disease"]
            drug_input = train_mask["drug"]
            labels = train_mask["label"]
            num_batch = int(math.ceil(len(disease_input) / batch_size))
            self.batch_data = list()

            for i in range(num_batch):
                input_disease = disease_input[i * batch_size: (i + 1) * batch_size]
                input_drug = drug_input[i * batch_size: (i + 1) * batch_size]
                label = labels[i * batch_size: (i + 1) * batch_size]
                self.batch_data.append(
                    [disease_drug_Adj, disease_disease_sim_Matrix, drug_drug_sim_Matrix, input_disease, input_drug,
                     label])

        elif type == "valid" or type == "test":
            disease_disease_sim_Matrix, drug_drug_sim_Matrix, test_mask, truth_label = data
            disease_input = test_mask["disease"]
            drug_input = test_mask["drug"]
            labels = test_mask["label"]
            num_batch = int(math.ceil(len(disease_input) / batch_size))
            self.batch_data = list()
            for i in range(num_batch):
                input_disease = disease_input[i * batch_size: (i + 1) * batch_size]
                input_drug = drug_input[i * batch_size: (i + 1) * batch_size]
                label = labels[i * batch_size: (i + 1) * batch_size]
                self.batch_data.append(
                    [disease_drug_Adj, disease_disease_sim_Matrix, drug_drug_sim_Matrix, input_disease, input_drug,
                     label])

        self.len_data = len(self.batch_data)

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


if __name__ == "__main__":
    config = config_model()
    config["simulate"] = False
    logger = get_logger("./temp")
    data_reader(logger, config, "Fdataset")
