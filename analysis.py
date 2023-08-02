from operator import itemgetter
from utils import get_logger, create_model
import os
import tensorflow as tf
from model import Model
import evaluation
from loader import *

def results_writer(specific_name):
    path = "./data/Fdataset/Fdataset"
    _, _, _, _, disease_name, _, _ = load_mat(path + ".mat")
    score = scio.loadmat('result/analysis/Fdataset/{}.mat'.format(specific_name))
    y_score = score["predict"].reshape(-1)
    for i in range(len(disease_name)):
        disease_name[i] = disease_name[i][0]
    disease2score = {}
    for i in range(len(y_score)):
        disease2score[disease_name[i]] = y_score[i]
    sorted_data = sorted(disease2score.items(), key=lambda x: x[1], reverse=True)[:30]
    sorted_candidate_disease = {}
    for i in range(len(sorted_data)):
        sorted_candidate_disease[sorted_data[i][0]] = sorted_data[i][1]

    path = os.path.join(".", "result/analysis/Fdataset/")
    if not os.path.exists(path):
        os.makedirs(path)
    candidate_drug_file = "{}-candidate-disease.txt".format(specific_name)
    with open(path + candidate_drug_file, "w") as f:
        for key, value in sorted_candidate_disease.items():
            f.write(str(key) + "\t" + "{}".format(value) + "\n")


def analysis_results(config):
    log_path = os.path.join(".", config['log_path'])
    logger = get_logger(log_path)

    mat = scio.loadmat("./data/Fdataset/Fdataset.mat")
    drug_sim = mat["drug"].astype(np.float)
    drug_name = mat["Wrname"].reshape(-1)

    drug_name2id = {}
    for i in range(drug_name.shape[0]):
        drug_name2id[drug_name[i][0]] = i
    new_drug_sim = np.copy(drug_sim)
    for i in range(new_drug_sim.shape[0]):
        new_drug_sim[i][i] = 0
    triplets = list()
    for i in range(new_drug_sim.shape[0]):
        for j in range(new_drug_sim.shape[1]):
            triplets.append((drug_name[i], drug_name[j], new_drug_sim[i][j]))
    sorted_triplets = sorted(triplets, key=itemgetter(2), reverse=True)
    top_triplets = sorted_triplets[:10]
    drug_set = set()
    for i in range(len(top_triplets)):
        drug_set.add(top_triplets[i][0][0])
        drug_set.add(top_triplets[i][1][0])
    drug_list = list(drug_set)
    drug_id_list = list()
    for i in range(len(drug_list)):
        drug_id_list.append(drug_name2id[drug_list[i]])

    for id in drug_id_list:
        config["specific_name"] = drug_name[id][0]
        disease_disease_sim_Matrix, drug_drug_sim_Matrix, train_matrix, all_train_mask, \
        all_test_mask = analysis_data_reader(config=config, logger=logger, specific_id=id, dataset=config['dataset'],
                                             disease_disease_topk=config['disease_knn_number'],
                                             drug_drug_topk=config['drug_knn_number'])
        all_scores, all_labels = [], []
        for fold_num in range(len(all_train_mask)):
            tf.reset_default_graph()
            tf.set_random_seed(config['seed'])
            np.random.seed(config['seed'])
            train_data = (disease_disease_sim_Matrix, drug_drug_sim_Matrix, all_train_mask[fold_num], train_matrix)
            valid_data = (disease_disease_sim_Matrix, drug_drug_sim_Matrix, all_test_mask[fold_num], train_matrix)
            test_data = (disease_disease_sim_Matrix, drug_drug_sim_Matrix, all_test_mask[fold_num], train_matrix)
            train_manager = analysis_BatchManager(train_data, config['batch_size'], "train")
            valid_manager = analysis_BatchManager(valid_data, config['batch_size'], 'valid')
            test_manager = analysis_BatchManager(test_data, config['batch_size'], "test")
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            steps_per_epoch = train_manager.len_data
            with tf.Session(config=tf_config) as sess:
                ckptpath = "ckpt/{}/{}-fold{}/".format(config['dataset'], config['dataset'], fold_num + 1)
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
                    train_manager = analysis_BatchManager(train_data, config['batch_size'], "train")
                    auroc, aupr = evaluate(sess, model, "valid", valid_manager, logger, all_scores, all_labels, config,
                                           fold_num)
                    print("fold {} valid auroc :{:>.5f}".format(fold_num + 1, auroc))
                    print("fold {} valid aupr :{:>.5f}".format(fold_num + 1, aupr))
                # save_model(sess, model, ckptpath, logger)
                final_test_auroc, final_test_aupr = evaluate(sess, model, "test", test_manager, logger,
                                                             all_scores,
                                                             all_labels, config, fold_num)
                print("fold {} final test auroc :{:>.5f}".format(fold_num + 1, final_test_auroc))
                print("fold {} final test aupr :{:>.5f}".format(fold_num + 1, final_test_aupr))

        results_writer(config['specific_name'])


def evaluate(sess, model, name, data, logger, all_scores, all_labels, config, fold_num=0):
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
        all_scores.append(scores)
        all_labels.append(labels)
        folder = os.path.join('result', "analysis", "Fdataset")
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_name = "{}.mat".format(config["specific_name"])
        file_path = os.path.join(folder, file_name)
        scio.savemat(file_path, {'label': labels,
                                 'predict': scores})
        return auroc, aupr
