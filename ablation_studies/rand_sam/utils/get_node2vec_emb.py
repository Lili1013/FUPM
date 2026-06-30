# import os
# import sys
# curPath = os.path.abspath(os.path.dirname((__file__)))
# rootPath = os.path.split(curPath)[0]
# PathProject = os.path.split(rootPath)[0]
# sys.path.append(rootPath)
# sys.path.append(PathProject)
import os

import os
import sys

curPath = os.path.abspath(os.path.dirname((__file__)))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

import pandas as pd
import pickle
import random
import torch
import numpy as np
from loguru import logger
import networkx as nx
from gensim.models.doc2vec import Doc2Vec
from node2vec import Node2Vec
from gensim.models import Word2Vec


from para_parser import parse
args = parse()


class Load_Data(object):

    def __init__(self, **params):
        self.rating_file_path = params['rating_path']
        self.samplling_probability = params['samplling_probability']
        self.model_D2V_user = params['model_D2V_user']
        self.model_D2V_item = params['model_D2V_item']
        # self.user_rev_emb_path = params['user_rev_emb_path']
        # self.item_rev_emb_path = params['item_rev_emb_path']
        # self.node2vec_emb_path = params['node2vec_emb_path']
        # # self.train_path = params['train_path']
        # # self.test_path = params['test_path']
        # self.overlap_user_num = params['overlap_user_num']
        # self.overlap_users = [i for i in range(self.overlap_user_num)]
        self.graph = nx.Graph()
        self.load_orig_dataset()
        self.split_datasets()
        # self.create_data_loader()
        if self.model_D2V_user:
            self.adj = self.getAdj()
        else:
            self.adj = None

    def load_orig_dataset(self):
        self.df_rating = pd.read_csv(self.rating_file_path)[['userID', 'itemID', 'rating']]
        self.df_rating['category'] = 0
        self.num_users = len(self.df_rating['userID'].unique())
        self.num_items = len(self.df_rating['itemID'].unique())


    def split_datasets(self):
        '''
            split train data and test data
            :param df:
            :return:
            '''
        train = []
        test = []
        for x in self.df_rating.groupby(by='userID'):
            # test_item = random.choice(list(x[1]['itemID']))
            test_item = list(x[1]['itemID'])[-1]
            each_test = x[1][x[1]['itemID'].isin([test_item])][['userID', 'itemID', 'category', 'rating']]
            test.append(each_test)
            items = list(x[1]['itemID'])
            train_items = list(set(items).difference(set([test_item])))
            each_train = x[1][x[1]['itemID'].isin(train_items)][['userID', 'itemID', 'category', 'rating']]
            train.append(each_train)

        self.train_df = pd.concat(train, axis=0, ignore_index=True)
        # self.train_df.to_csv(self.train_path, index=False)
        logger.info('the number of train data is {}'.format(len(self.train_df)))

        self.test_df = pd.concat(test, axis=0, ignore_index=True)
        # self.test_df.to_csv(self.test_path, index=False)
        logger.info('the number of test data is {}'.format(len(self.test_df)))

    def getAdj(self):
        n_allnodes = self.num_users + self.num_items  # the number of all nodes (users and items)
        adj = np.zeros([1, n_allnodes, n_allnodes], dtype=np.float32)
        # User-item interactions
        for i in self.train_df.itertuples():
            user = i.userID
            item = i.itemID
            rating = i.rating
            weight = rating / 5
            adj[0][user][self.num_users + item] = 1  # [0,self.shape[0]-1]: users, [self.shape[0],n_allnodes-1]: items
            adj[0][self.num_users + item][user] = 1
            self.graph.add_weighted_edges_from([(user, self.num_users + item, weight)])

        for i in range(self.num_users):
            for j in range(i + 1, self.num_users):
                sim = self.model_D2V_user.docvecs.similarity(i, j)
                rand = random.uniform(0, 1)
                if rand < sim * self.samplling_probability:
                    adj[0][i][j] = 1
                    adj[0][j][i] = 1
                    self.graph.add_weighted_edges_from([(i, j, sim)])

        for i in range(self.num_items):
            for j in range(i + 1, self.num_items):
                sim = self.model_D2V_item.docvecs.similarity(i, j)
                rand = random.uniform(0, 1)
                if rand < sim * self.samplling_probability:
                    adj[0][i+self.num_users][j+self.num_users] = 1
                    adj[0][j+self.num_users][i+self.num_users] = 1
                    self.graph.add_weighted_edges_from([(i, j, sim)])

        return adj


def get_node2vec_emb(data_path,tag_name,model_path,emb_save_path):
    df = pd.read_csv(data_path)
    user_num = len(df['userID'].unique())
    model = Word2Vec.load(model_path)
    logger.info('obtain user embeddings')
    user_embeddings = {}
    for index, row in df.iterrows():
        user_id = row[tag_name]
        if tag_name == 'itemID':
            user_embedding = model.wv[user_id+user_num]
        else:
            user_embedding = model.wv[user_id]
        user_embeddings[user_id] = user_embedding
    #
    # # save user_embeddings
    logger.info('save user embeddings')
    # with open(emb_save_path, 'wb') as file:
    #     pickle.dump(user_embeddings, file)
    all_vectors = []
    for key, value in user_embeddings.items():
        all_vectors.append(value)
    vectors_array = np.array(all_vectors)
    np.save(emb_save_path, vectors_array)

def pickle_to_npy(source_path,to_path):
    all_vectors = []
    with open(source_path, 'rb') as f:
        text_feat = pickle.load(f)
    for key, value in text_feat.items():
        all_vectors.append(value)
    vectors_array = np.array(all_vectors)
    np.save(to_path, vectors_array)


if __name__ == '__main__':
    # model_D2V_user = Doc2Vec.load('../datasets/debug_datasets/douban_book/doc_models/book_user_doc_model_32.model')
    # model_D2V_item = Doc2Vec.load('../datasets/debug_datasets/douban_book/doc_models/book_item_doc_model_32.model')
    #
    # data_params = {
    #     'rating_path': '../datasets/debug_datasets/douban_book/book_review_data_new.csv',
    #     'model_D2V_user':model_D2V_user,
    #     'model_D2V_item':model_D2V_item,
    #     'samplling_probability':0.5,
    # }
    # load_data = Load_Data(**data_params)
    # graph = load_data.graph
    # logger.info('start train node2vec')
    # node2vec = Node2Vec(graph, dimensions=32, walk_length=30, num_walks=100, workers=1)
    # model_N2V = node2vec.fit(window=10, min_count=1, batch_words=4)
    # model_N2V.save('../datasets/debug_datasets/douban_book/doc_models/book_node2vec_model_32.model')
    #
    # model_D2V_user = Doc2Vec.load('../datasets/debug_datasets/douban_movie/doc_models/movie_user_doc_model_32.model')
    # model_D2V_item = Doc2Vec.load('../datasets/debug_datasets/douban_movie/doc_models/movie_item_doc_model_32.model')
    #
    # data_params = {
    #     'rating_path': '../datasets/debug_datasets/douban_movie/movie_review_data_new.csv',
    #     'model_D2V_user': model_D2V_user,
    #     'model_D2V_item': model_D2V_item,
    #     'samplling_probability': 0.5,
    # }
    #
    # load_data = Load_Data(**data_params)
    # graph = load_data.graph
    # logger.info('start train node2vec')
    # node2vec = Node2Vec(graph, dimensions=32, walk_length=30, num_walks=100, workers=4)
    # model_N2V = node2vec.fit(window=10, min_count=1, batch_words=4)
    # model_N2V.save('../datasets/debug_datasets/douban_movie/doc_models/movie_node2vec_model_32.model')
    # #
    # model_D2V_user = Doc2Vec.load('../datasets/debug_datasets/douban_music/doc_models/music_user_doc_model_32')
    # model_D2V_item = Doc2Vec.load('../datasets/debug_datasets/douban_music/doc_models/music_item_doc_model_32')
    #
    # data_params = {
    #     'rating_path': '../datasets/debug_datasets/douban_music/music_review_data_new.csv',
    #     'model_D2V_user': model_D2V_user,
    #     'model_D2V_item': model_D2V_item,
    #     'samplling_probability': 0.5,
    # }
    #
    # load_data = Load_Data(**data_params)
    # graph = load_data.graph
    # logger.info('start train node2vec')
    # node2vec = Node2Vec(graph, dimensions=32, walk_length=30, num_walks=100, workers=1)
    # model_N2V = node2vec.fit(window=10, min_count=1, batch_words=4)
    # model_N2V.save('../datasets/debug_datasets/douban_music/doc_models/music_node2vec_model_32.model')
    # get_node2vec_emb(data_path='../datasets/debug_datasets/douban_book/book_review_data_new.csv',tag_name='userID',
    #                  model_path='../datasets/debug_datasets/douban_book/doc_models/book_node2vec_model_32.model',
    #                  emb_save_path='../datasets/debug_datasets/douban_book/doc_embs/book_user_nod_emb_32.npy')
    # get_node2vec_emb(data_path='../datasets/debug_datasets/douban_book/book_review_data_new.csv',tag_name='itemID',
    #                  model_path='../datasets/debug_datasets/douban_book/doc_models/book_node2vec_model_32.model',
    #                  emb_save_path='../datasets/debug_datasets/douban_book/doc_embs/book_item_nod_emb_32.npy')
    #
    #
    # get_node2vec_emb(data_path='../datasets/debug_datasets/douban_movie/movie_review_data_new.csv', tag_name='userID',
    #                  model_path='../datasets/debug_datasets/douban_movie/doc_models/movie_node2vec_model_32.model',
    #                  emb_save_path='../datasets/debug_datasets/douban_movie/doc_embs/movie_user_nod_emb_32.npy')
    # get_node2vec_emb(data_path='../datasets/debug_datasets/douban_movie/movie_review_data_new.csv', tag_name='itemID',
    #                  model_path='../datasets/debug_datasets/douban_movie/doc_models/movie_node2vec_model_32.model',
    #                  emb_save_path='../datasets/debug_datasets/douban_movie/doc_embs/movie_item_nod_emb_32.npy')
    #
    # get_node2vec_emb(data_path='../datasets/debug_datasets/douban_music/music_review_data_new.csv', tag_name='userID',
    #                  model_path='../datasets/debug_datasets/douban_music/doc_models/music_node2vec_model_32.model',
    #                  emb_save_path='../datasets/debug_datasets/douban_music/doc_embs/music_user_nod_emb_32.npy')
    # get_node2vec_emb(data_path='../datasets/debug_datasets/douban_music/music_review_data_new.csv', tag_name='itemID',
    #                  model_path='../datasets/debug_datasets/douban_music/doc_models/music_node2vec_model_32.model',
    #                  emb_save_path='../datasets/debug_datasets/douban_music/doc_embs/music_item_nod_emb_32.npy')








    # model_D2V_user = Doc2Vec.load('/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_book/doc_models/book_user_doc_model_32')
    # model_D2V_item = Doc2Vec.load('/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_book/doc_models/book_item_doc_model_32')
    #
    # data_params = {
    #     'rating_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_book/book_review_data_new.csv',
    #     'model_D2V_user': model_D2V_user,
    #     'model_D2V_item': model_D2V_item,
    #     'samplling_probability': 0.5,
    # }
    # load_data = Load_Data(**data_params)
    # graph = load_data.graph
    # logger.info('start train node2vec')
    # node2vec = Node2Vec(graph, dimensions=32, walk_length=30, num_walks=100, workers=1)
    # model_N2V = node2vec.fit(window=10, min_count=1, batch_words=4)
    # model_N2V.save('../datasets/book_node2vec_model_32.model')
    # logger.info('end')

    model_D2V_user = Doc2Vec.load("/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/"
                                          "amazon_phone_sport/amazon_phone/doc_models/phone_user_doc_model_64.model")
    model_D2V_item = Doc2Vec.load("/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/"
                                          "amazon_phone_sport/amazon_phone/doc_models/phone_item_doc_model_64.model")

    data_params = {
        'rating_path': '/data/lwang9/CDR_data_process/amazon_data_process_FedPCL_MDR/datasets/amazon_phone_sport/amazon_phone/phone_data.csv',
        'model_D2V_user': model_D2V_user,
        'model_D2V_item': model_D2V_item,
        'samplling_probability': 0.5,
    }

    load_data = Load_Data(**data_params)
    graph = load_data.graph
    logger.info('start train node2vec')
    node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=100, workers=1)
    model_N2V = node2vec.fit(window=10, min_count=1, batch_words=4)
    model_N2V.save('../datasets/phone_node2vec_model.model')
    logger.info('end')


    # model_D2V_user = Doc2Vec.load('/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_music/doc_models/music_user_doc_model_32')
    # model_D2V_item = Doc2Vec.load('/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_music/doc_models/music_item_doc_model_32')
    #
    # data_params = {
    #     'rating_path': '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_music/music_review_data_new.csv',
    #     'model_D2V_user': model_D2V_user,
    #     'model_D2V_item': model_D2V_item,
    #     'samplling_probability': 0.5
    # }
    #
    # load_data = Load_Data(**data_params)
    # graph = load_data.graph
    # logger.info('start train node2vec')
    # node2vec = Node2Vec(graph, dimensions=32, walk_length=30, num_walks=100, workers=1)
    # model_N2V = node2vec.fit(window=10, min_count=1, batch_words=4)
    # model_N2V.save('../datasets/music_node2vec_model_32.model')
    # logger.info('end')

