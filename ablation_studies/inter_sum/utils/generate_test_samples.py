import pickle

import pandas as pd
import numpy as np

import multiprocessing
from functools import partial
from loguru import logger
import random

# def generate_test_negatives(df_all,df_test,df_train,test_neg_num,test_pos_neg_path):
#     '''
#     generate test positive and negative samples including one positive and 99 negative samples
#     :return:
#     '''
#     negatives = []
#     num_negatives = test_neg_num
#     # df = data[['userID', 'itemID', 'rating']]
#     num_items = len(df_all['itemID'].unique())
#     for index, row in df_test.iterrows():
#         print(index)
#         each_negatives = []
#         uid = row['userID']
#         iid = row['itemID']
#         each_negatives.append(int(uid))
#         each_negatives.append(int(iid))
#         for t in range(num_negatives):
#             j = np.random.randint(num_items)
#             while (len(df_all[(df_all['userID'].isin([uid])) & (df_all['itemID'].isin([j]))]) > 0) \
#                     or (len(df_train[(df_train['userID'].isin([uid])) & (df_train['itemID'].isin([j]))]) > 0):
#                 j = np.random.randint(num_items)
#             each_negatives.append(j)
#         negatives.append(each_negatives)
#     print('start store')
#     with open(test_pos_neg_path, 'w') as f:
#         for each_list in negatives:
#             for item in each_list:
#                 f.write(str(item) + " ")
#             f.write("\n")


def generate_test_negatives(index_range, df_all, df_test, df_train, test_neg_num, num_items):
    negatives = []
    num_negatives = test_neg_num
    i = index_range[0]
    while i < index_range[1]:
        each_negatives = []
        row = df_test.iloc[i]
        uid = row['userID']
        iid = row['itemID']

        each_negatives.append(int(uid))
        each_negatives.append(int(iid))

        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (len(df_all[(df_all['userID'].isin([uid])) & (df_all['itemID'].isin([j]))]) > 0) \
                    or (len(df_train[(df_train['userID'].isin([uid])) & (df_train['itemID'].isin([j]))]) > 0):
                j = np.random.randint(num_items)
            each_negatives.append(j)
        i+=1
        logger.info(i)

        negatives.append(each_negatives)

    return negatives

def generate_and_store_test_negatives(df_all, df_test, df_train, test_neg_num, test_pos_neg_path):
    logger.info('start generate samples')
    num_items = len(df_all['itemID'].unique())
    num_samples = len(df_test)
    chunk_size = num_samples // multiprocessing.cpu_count()
    index_ranges = [(i * chunk_size, min((i + 1) * chunk_size, num_samples)) for i in range(multiprocessing.cpu_count())]
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    func = partial(generate_test_negatives, df_all=df_all, df_test=df_test, df_train=df_train, test_neg_num=test_neg_num, num_items=num_items)
    results = pool.map(func, index_ranges)
    pool.close()
    pool.join()
    if index_ranges[-1][1] < len(df_test):
        index_ranges = (index_ranges[-1][1],len(df_test))
        logger.info('gggggggggggg')
        negatives = generate_test_negatives(index_ranges, df_all, df_test, df_train, test_neg_num, num_items)
        results.append(negatives)
    logger.info('start write samples')
    with open(test_pos_neg_path, 'w') as f:
        for each_list in results:
            for items in each_list:
                for item in items:
                    f.write(str(item) + " ")
                f.write("\n")

if __name__ == '__main__':
    dataset = 'amazon_sport_cloth_phone'
    domain = 'phone'
    path = f'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/{dataset}/{domain}/{domain}_inter.csv'
    train_path = f'../../datasets/{dataset}/{domain}/train.csv'
    test_path = f'../../datasets/{dataset}/{domain}/test.csv'
    to_path = f'../../datasets/{dataset}/{domain}/test.txt'
    # dataset = 'douban_book'
    # path = '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/{}/{}_review_data_new.csv'.format(dataset,dataset.split('_')[1])
    # train_path = '../datasets/{}/{}/train.csv'.format(dataset.split('_')[0],dataset.split('_')[1])
    # test_path = '../datasets/{}/{}/test.csv'.format(dataset.split('_')[0],dataset.split('_')[1])
    # to_path = '../datasets/{}/{}/test.txt'.format(dataset.split('_')[0],dataset.split('_')[1])
    df = pd.read_csv(path)
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    print(len(test_data))
    generate_and_store_test_negatives(df, test_data, train_data, 99,to_path)
