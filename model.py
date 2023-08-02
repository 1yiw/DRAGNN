# Graph Neural Networks for Drug Repositioning
import tensorflow as tf
import numpy as np
from keras.regularizers import l2
from tensorflow.python.keras.layers import Dense
from utils import random_uniform_init
from clr import cyclic_learning_rate


class Model(object):

    def __init__(self, config):
        """
        :param config:
        """
        self.config = config
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.disease_dim = config['disease_dim']
        self.drug_dim = config['drug_dim']
        self.disease_size = config['disease_size']
        self.drug_size = config['drug_size']
        self.latent_dim = config['latent_dim']
        self.attention_flag = config['attention_flag']
        self.atten_dim = config['atten_dim']
        self.l2 = config['l2']  # init=0

        self.global_step = tf.Variable(0, trainable=False)
        # self.best_dev_auroc = tf.Variable(0.0, trainable=False)
        # self.best_test_auroc = tf.Variable(0.0, trainable=False)
        # self.best_dev_aupr = tf.Variable(0.0, trainable=False)
        # self.best_test_aupr = tf.Variable(0.0, trainable=False)

        # input
        self.e_p_Adj = tf.placeholder(dtype=tf.float32,
                                      shape=[self.disease_size, self.drug_size])
        self.e_e_Adj = tf.placeholder(dtype=tf.float32,
                                      shape=[self.disease_size, self.disease_size])
        self.p_p_Adj = tf.placeholder(dtype=tf.float32,
                                      shape=[self.drug_size, self.drug_size])

        self.input_disease = tf.placeholder(dtype=tf.int32, shape=[None])
        self.input_drug = tf.placeholder(dtype=tf.int32, shape=[None])
        self.label = tf.placeholder(dtype=tf.float32, shape=[None])

        self.disease_embedding = random_uniform_init(name="disease_embedding_matrix",
                                                    shape=[self.disease_size, self.disease_dim])
        self.drug_embedding = random_uniform_init(name="drug_embedding_matrix",
                                                      shape=[self.drug_size, self.drug_dim])

        with tf.variable_scope("model_disease", reuse=tf.AUTO_REUSE):
            gcn_output = self.gcn(self.e_p_Adj, self.drug_embedding, self.drug_dim, self.disease_embedding,
                                  self.latent_dim, self.attention_flag)

            # Modeling top-k diseases Interaction information
            if config['disease_knn_number'] > 0:
                disease_edges = tf.reduce_sum(self.e_e_Adj, 1)
                disease_edges = tf.tile(tf.expand_dims(disease_edges, 1), [1, self.disease_dim])
                ave_disease_edges = tf.divide(tf.matmul(self.e_e_Adj, self.disease_embedding), disease_edges)

                w2 = tf.get_variable('w2', shape=[self.disease_dim, self.latent_dim],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
                b2 = tf.get_variable('b2', shape=[self.latent_dim],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
                h_e_e = tf.nn.xw_plus_b(ave_disease_edges, w2, b2)  # disease_size*latent_dim

                self.h_e = tf.nn.selu(tf.add(gcn_output, h_e_e))
            else:
                self.h_e = tf.nn.selu(gcn_output)

        with tf.variable_scope("model_drug", reuse=tf.AUTO_REUSE):
            gcn_output = self.gcn(tf.transpose(self.e_p_Adj), self.disease_embedding, self.disease_dim,
                                  self.drug_embedding, self.latent_dim, self.attention_flag)

            # Modeling top-k drugs Interaction information
            if config['drug_knn_number'] > 0:
                drug_edges = tf.reduce_sum(self.p_p_Adj, 1)
                drug_edges = tf.tile(tf.expand_dims(drug_edges, 1), [1, self.drug_dim])
                ave_drug_edges = tf.divide(tf.matmul(self.p_p_Adj, self.drug_embedding), drug_edges)

                w3 = tf.get_variable('w3', shape=[self.drug_dim, self.latent_dim],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
                b3 = tf.get_variable('b3', shape=[self.latent_dim],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1))
                h_p_p = tf.nn.xw_plus_b(ave_drug_edges, w3, b3)  # drug_size*latent_dim

                self.h_p = tf.nn.selu(tf.add(gcn_output, h_p_p))
            else:
                self.h_p = tf.nn.selu(gcn_output)

        with tf.variable_scope("drug_rec", reuse=tf.AUTO_REUSE):
            h_e_1 = tf.nn.embedding_lookup(self.h_e, self.input_disease)  # batch_size * disease_latent_dim
            h_p_1 = tf.nn.embedding_lookup(self.h_p, self.input_drug)  # batch_size * drug_latent_dim
            input_temp = tf.multiply(h_e_1, h_p_1)
            for l_num in range(config['mlp_layer_num']):
                input_temp = Dense(self.disease_dim, activation='selu', kernel_initializer='lecun_uniform')(input_temp)  # MLP hidden layer
            z = Dense(1, kernel_initializer='lecun_uniform', name='prediction')(input_temp)
            z = tf.squeeze(z)

        self.label = tf.squeeze(self.label)
        self.loss = tf.losses.sigmoid_cross_entropy(self.label, z)
        self.z = tf.sigmoid(z)

        # train
        with tf.variable_scope("optimizer"):
            self.opt = tf.train.AdamOptimizer(learning_rate=cyclic_learning_rate(global_step=self.global_step,
                                                                                 learning_rate=self.lr*0.1,
                                                                                 max_lr=self.lr,
                                                                                 mode='exp_range',
                                                                                 gamma=.999))
            # apply grad clip to avoid gradient explosion
            self.grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in self.grads_vars]

            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def gcn(self, adj, ner_inputs, ner_dim, self_inputs, latent_dim, attention_flag=False):
        """
        Aggregate information from neighbor nodes
        :param adj: Adjacency matrix
        :param attention_flag: GAT flag
        :param ner_inputs: disease or drug embedding
        :param ner_dim: ner_inputs dimension
        :param self_inputs:
        :param latent_dim: output dimension
        :return:
        """
        # aggregate heterogeneous information
        if attention_flag:
            query = tf.tile(tf.reshape(self_inputs, (self_inputs.shape[0], 1, self_inputs.shape[1])), [1, ner_inputs.shape[0], 1])
            key = tf.tile(tf.reshape(ner_inputs, (1, ner_inputs.shape[0], ner_inputs.shape[1])), [self_inputs.shape[0], 1, 1])
            key_query = tf.reshape(tf.concat([key, query], -1), [ner_inputs.shape[0]*self_inputs.shape[0], -1])
            alpha = Dense(self.atten_dim, activation='relu', use_bias=True, kernel_regularizer=l2(self.l2))(key_query)
            alpha = Dense(1, activation='relu', use_bias=True, kernel_regularizer=l2(self.l2))(alpha)
            alpha = tf.reshape(alpha, [self_inputs.shape[0], ner_inputs.shape[0]])
            alpha = tf.multiply(alpha, adj)  # disease_size * drug_size
            alpha_exps = tf.nn.softmax(alpha, 1)
            w1 = tf.get_variable('w1', shape=[ner_dim, latent_dim],
                                 initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
            b1 = tf.get_variable('b1', shape=[latent_dim],
                                 initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
            alpha_exps = tf.tile(tf.expand_dims(alpha_exps, -1), [1, 1, ner_inputs.shape[1]])
            e_r = tf.nn.xw_plus_b(tf.reduce_sum(tf.multiply(alpha_exps, key), 1), w1, b1)
        else:
            edges = tf.matmul(adj, ner_inputs)
            w1 = tf.get_variable('w1', shape=[ner_dim, latent_dim],
                                 initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
            b1 = tf.get_variable('b1', shape=[latent_dim],
                                 initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
            e_r = tf.nn.xw_plus_b(edges, w1, b1)

        return e_r

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        disease_drug_Adj, disease_disease_Adj, drug_drug_Adj, input_disease, input_drug, label = batch
        feed_dict = {
            self.e_p_Adj: np.asarray(disease_drug_Adj),
            self.e_e_Adj: np.asarray(disease_disease_Adj),
            self.p_p_Adj: np.asarray(drug_drug_Adj),
            self.input_disease: np.asarray(input_disease),
            self.input_drug: np.asarray(input_drug),
            self.label: np.asarray(label)
        }
        if is_train:
            global_step, loss, z, grads_vars, _ = sess.run(
                [self.global_step, self.loss, self.z, self.grads_vars, self.train_op], feed_dict)
            return global_step, loss, z, grads_vars
        else:
            z, labels = sess.run([self.z, self.label], feed_dict)
            return z, labels
