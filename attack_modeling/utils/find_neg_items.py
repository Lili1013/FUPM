import pandas as pd
import pickle
import torch
import numpy as np
import torch.nn.functional as F

import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

def contruct_sim_matrx(embeddings):
    # # 加载嵌入向量字典
    # embeddings_dict = np.load('item_embeddings.npy', allow_pickle=True).item()
    #
    # # 提取嵌入向量
    # item_ids = list(embeddings_dict.keys())
    # embeddings = np.array([embeddings_dict[item_id] for item_id in item_ids])

    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(embeddings)

    # 设定相似度阈值并转换为稀疏矩阵
    threshold = 0.0
    row, col = np.where(similarity_matrix > threshold)
    data = similarity_matrix[row, col]
    sparse_similarity_matrix = sp.coo_matrix((data, (row, col)), shape=similarity_matrix.shape)

    # 输出稀疏矩阵的一些信息
    print(f'Sparse matrix shape: {sparse_similarity_matrix.shape}')
    print(f'Number of non-zero elements: {sparse_similarity_matrix.nnz}')
    return sparse_similarity_matrix


def find_negatives(df, similarity_matrix, threshold=0.5):
    # 初始化潜在正样本的字典
    potential_negatives = {}

    # 获取所有唯一的用户和项目
    users = df['userID'].unique()
    all_items = set(df['itemID'].unique())
    item_id_to_index = {item_id:idx for idx,item_id in enumerate(sorted(all_items))}

    # 遍历每个用户
    i = 0
    for user in users:
        if i % 1000 == 0:
            logger.info(i)
        i += 1
        interacted_items = df[df['userID'] == user]['itemID'].unique()
        non_interacted_items = list(all_items - set(interacted_items))

        if not non_interacted_items:
            continue

        # 获取用户已交互项目的索引
        interacted_indices = [item_id_to_index[item] for item in interacted_items if item in item_id_to_index]
        non_interacted_indices = [item_id_to_index[item] for item in non_interacted_items if item in item_id_to_index]

        sim_scores_matrix = similarity_matrix[non_interacted_indices][:,interacted_indices]
        avg_sim_scores = sim_scores_matrix.mean(axis=1).A.flatten()

        # # 对于每个未交互项目，计算与所有已交互项目的平均相似度
        # scores = []
        # for item in non_interacted_items:
        #     item_idx = item_id_to_index[item]
        #     sim_scores = similarity_matrix[item_idx, interacted_indices].toarray().flatten()
        #     avg_sim_score = np.mean(sim_scores)
        #     scores.append((item, avg_sim_score))

        # 筛选出平均相似度高于阈值的项目
        # potential_items = [item for item, score in scores if score > threshold]
        potential_items = [non_interacted_items[idx] for idx,score in enumerate(avg_sim_scores) if score<threshold]
        potential_negatives[user] = potential_items

    return potential_negatives


if __name__ == '__main__':
    dataset = 'douban_book_music'
    domain = 'music'
    # df = pd.read_csv(f'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/{dataset}/{domain}/{domain}_inter.csv')
    df = pd.read_csv(
        f'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/{dataset}/{domain}/{domain}_review_data.csv')
    # df['itemID'] = df['itemID'].astype(int)
    with open(f'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/{dataset}/{domain}/{domain}_item_doc_emb_sbert.npy', 'rb') as f_user:
        phone_item_review_emb = torch.from_numpy(np.load(f_user))
    # with open(f'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/{dataset}/{domain}/{domain}_item_doc_emb_128.npy', 'rb') as f_user:
    #     phone_item_review_emb = torch.from_numpy(np.load(f_user))
    print('start cal sim')
    item_sim_matrix = contruct_sim_matrx(phone_item_review_emb)
    item_sim_matrix = item_sim_matrix.tocsr()
    print('start find potential items')
    potential_positives = find_negatives(df,item_sim_matrix,0.5)
    print('start save')
    with open(f'../../datasets/{dataset}/{domain}/user_neg_items_doc_emb_sbert_0.5.pkl','wb') as f:
        pickle.dump(potential_positives,f)



