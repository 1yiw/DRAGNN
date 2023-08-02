import argparse
import os
from collections import OrderedDict
import tensorflow as tf
import numpy as np
import sys
import pickle
from loader import data_reader, BatchManager
from model import Model
from utils import get_logger, create_model, save_model
import evaluation
import random
from case_study import case_study
from analysis import analysis_results


sys.path.append(".")

def parse_args():
    parser = argparse.ArgumentParser(description="Run DRGNN.")
    parser.add_argument('--dataset', nargs='?', default='Fdataset', help='Choose a dataset. [Fdataset/Cdataset/LRSSL]')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024*3, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='init Learning rate.')
    parser.add_argument('--mode', type=str, default='cv', help='cv, case, analysis.')
    parser.add_argument('--disease_dim', type=int, default=125)
    parser.add_argument('--drug_dim', type=int, default=125)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--atten_dim', type=int, default=64)
    parser.add_argument('--attention_flag', type=int, default=1)  # 0 is False, 1 is true
    parser.add_argument('--l2', type=float, default=1)
    parser.add_argument('--steps_check', type=int, default=20)
    parser.add_argument('--disease_knn_number', type=int, default=7)
    parser.add_argument('--drug_knn_number', type=int, default=7)
    parser.add_argument("--mlp_layer_num", type=int, default=1)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--specific_name", type=str, default='parkinson', help='choose parkinson or breast cancer.')
    parser.add_argument("--specific_id", type=int, default=0)
    return parser.parse_args()


specific_id = 0
args = parse_args()
if args.mode == "case":
    args.seed = 11
    if args.specific_name == "parkinson":
        specific_id = 119
    elif args.specific_name == "breast cancer":
        specific_id = 19

dataset = args.dataset
lr = args.lr
batch_size = args.batch_size
max_epoch = args.epochs
mode = args.mode
disease_dim = args.disease_dim
drug_dim = args.drug_dim
latent_dim = args.latent_dim
attention_flag = args.attention_flag
atten_dim = args.atten_dim
l2 = args.l2
steps_check = args.steps_check
disease_knn_number = args.disease_knn_number
drug_knn_number = args.drug_knn_number
mlp_layer_num = args.mlp_layer_num
seed = args.seed
n_splits = args.n_splits
specific_name = args.specific_name


def config_model():
    config = OrderedDict()
    config['dataset'] = dataset
    config['lr'] = lr
    config['batch_size'] = batch_size
    config['disease_dim'] = disease_dim
    config['drug_dim'] = drug_dim
    config['latent_dim'] = latent_dim
    config['attention_flag'] = (attention_flag == 1)
    config['atten_dim'] = atten_dim
    config["clip"] = 3
    config['log_path'] = "log/" + dataset
    config['max_epoch'] = max_epoch
    config['steps_check'] = steps_check
    config['l2'] = l2
    config['mode'] = mode
    config['disease_knn_number'] = disease_knn_number
    config['drug_knn_number'] = drug_knn_number
    config['mlp_layer_num'] = mlp_layer_num
    config['seed'] = seed
    config['n_splits'] = n_splits
    config['specific_name'] = specific_name
    config['specific_id'] = specific_id
    return config


config = config_model()
random.seed(seed)

def cross_validation():
    log_path = os.path.join(".", config['log_path'])
    logger = get_logger(log_path)
    map_file_path = "./pkl/" + dataset + "/data.pkl"
    if os.path.isfile(map_file_path):
        with open(map_file_path, "rb") as f:
            disease_disease_sim_Matrix, drug_drug_sim_Matrix, truth_label, \
            all_train_mask, all_test_mask = pickle.load(f)
    else:
        disease_disease_sim_Matrix, drug_drug_sim_Matrix, truth_label, \
        all_train_mask, all_test_mask = data_reader(logger,
                                                    config=config,
                                                    dataset=dataset,
                                                    disease_disease_topk=disease_knn_number,
                                                    drug_drug_topk=drug_knn_number)
    final_all_auroc, final_all_aupr = [], []
    for fold_num in range(len(all_train_mask)):
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)
        train_data = (disease_disease_sim_Matrix, drug_drug_sim_Matrix, all_train_mask[fold_num], truth_label)
        valid_data = (disease_disease_sim_Matrix, drug_drug_sim_Matrix, all_test_mask[fold_num], truth_label)
        test_data = (disease_disease_sim_Matrix, drug_drug_sim_Matrix, all_test_mask[fold_num], truth_label)
        train_manager = BatchManager(train_data, config['batch_size'], "train")
        valid_manager = BatchManager(valid_data, config['batch_size'], 'valid')
        test_manager = BatchManager(test_data, config['batch_size'], "test")
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        steps_per_epoch = train_manager.len_data
        with tf.Session(config=tf_config) as sess:
            ckptpath = "ckpt/{}/{}-fold{}/".format(dataset, dataset, fold_num + 1)
            model = create_model(sess, Model, ckptpath, config, logger)
            logger.info("start training fold {}".format(fold_num + 1))
            loss_list = []
            for i in range(config['max_epoch']):
                for batch in train_manager.iter_batch(shuffle=True):
                    step, loss, z, grads_vars = model.run_step(sess, True, batch)
                    loss_list.append(loss)
                    if step % config['steps_check'] == 0:
                        iteration = step // steps_per_epoch + 1
                        logger.info("epoch:{} step:{}/{}, loss:{:>9.6f}".format(
                            iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss_list)))
                        loss_list = []
                train_manager = BatchManager(train_data, config['batch_size'], "train")
                auroc, aupr = evaluate(sess, model, "valid", valid_manager, logger, fold_num)
                print("fold {} valid auroc :{:>.5f}".format(fold_num + 1, auroc))
                print("fold {} valid aupr :{:>.5f}".format(fold_num + 1, aupr))
            # save_model(sess, model, ckptpath, logger)
            final_test_auroc, final_test_aupr = evaluate(sess, model, "test", test_manager, logger, fold_num)
            final_all_auroc.append(final_test_auroc)
            final_all_aupr.append(final_test_aupr)
            print("fold {} final test auroc :{:>.5f}".format(fold_num + 1, final_test_auroc))
            print("fold {} final test aupr :{:>.5f}".format(fold_num + 1, final_test_aupr))
    print("final_avg_auroc :{:>.5f} final_avg_aupr :{:>.5f}".format(np.mean(final_all_auroc),
                                                                    np.mean(final_all_aupr)))



def evaluate(sess, model, name, data, logger, fold_num=0):
    logger.info("evaluate data:{}".format(name))
    scores, labels = [], []
    for batch in data.iter_batch():
        # disease_drug_Adj, disease_disease_sim_Matrix, drug_drug_sim_Matrix, input_disease, input_drug, _ = batch
        score, label = model.run_step(sess, False, batch)
        scores.append(score)
        labels.append(label)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    result = evaluation.evaluate(scores, labels)
    auroc = result['auroc']
    aupr = result['aupr']
    if name == "valid":
        return auroc, aupr
    elif name == "test":
        logger.info("fold {} final test auroc :{:>.5f}".format(fold_num + 1, auroc))
        logger.info("fold {} final test aupr :{:>.5f}".format(fold_num + 1, aupr))
        # np.savetxt("save_txt/scores{}".format(fold_num + 1), scores, delimiter=" ")
        # np.savetxt("save_txt/labels{}".format(fold_num + 1), labels, delimiter=" ")
        return auroc, aupr


if __name__ == "__main__":
    if mode == "cv":
        cross_validation()
    elif mode == "case":
        case_study(config)
    elif mode == "analysis":
        analysis_results(config)