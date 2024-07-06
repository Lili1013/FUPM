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


def find_potential_positives(df, similarity_matrix, threshold=0.5):
    # 初始化潜在正样本的字典
    potential_positives_dict = {}
    results = []

    # 获取所有唯一的用户和项目
    users = df['userID'].unique()
    all_items = set(df['itemID'].unique())
    item_id_to_index = {item_id:idx for idx,item_id in enumerate(sorted(all_items))}

    for index, row in df.iterrows():
        if index % 1000 == 0:
            logger.info(index)
        user_id = row['userID']
        item_id = row['itemID']

        # 找到用户未交互的items
        interacted_items = df[df['userID'] == user_id]['itemID'].values
        non_interacted_items = [item for item in all_items if item not in interacted_items]

        # # 找到item_id的索引
        # item_index = np.where(all_items == item_id)[0][0]

        # 计算相似度并找到potential positive items
        # non_interacted_indices = [np.where(all_items == item)[0][0] for item in non_interacted_items]
        potential_positives = similarity_matrix[item_id][:, non_interacted_items].toarray()[0]

        # 选择相似度大于0.5的items
        potential_positive_items = [non_interacted_items[i] for i in range(len(potential_positives)) if
                                    potential_positives[i] > threshold]
        potential_positives_dict[(user_id,item_id)] = potential_positive_items

        # # 将结果添加到列表中
        # results.append([user_id, item_id] + potential_positive_items)

    return potential_positives_dict


if __name__ == '__main__':
    # to_dataset = 'amazon_phone_sport_diff_inter_ratio'
    # to_sub_dataset = 'amazon_phone_sport_0.6'
    dataset = 'amazon_sport_cloth_phone'
    domain = 'cloth'
    # ratio = '0.6'
    # df = pd.read_csv(f'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/{dataset}/{domain}/{domain}_inter.csv')
    # df = df.sort_values(by='userID')
    df = pd.read_csv(
        f'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/{dataset}/{domain}/{domain}_inter.csv')
    # df['itemID'] = df['itemID'].astype(int)
    with open(f'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/{dataset}/{domain}/{domain}_item_doc_emb_256.npy', 'rb') as f_user:
        phone_item_review_emb = torch.from_numpy(np.load(f_user))
    # with open(f'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/{dataset}/{domain}/{domain}_item_doc_emb_128.npy', 'rb') as f_user:
    #     phone_item_review_emb = torch.from_numpy(np.load(f_user))
    print('start cal sim')
    item_sim_matrix = contruct_sim_matrx(phone_item_review_emb)
    item_sim_matrix = item_sim_matrix.tocsr()
    print('start find potential items')
    potential_positives = find_potential_positives(df,item_sim_matrix,0.5)
    print('start save')
    with open(f'../../datasets/{dataset}/{domain}/user_pos_items_doc_emb_sbert_0.5.pkl','wb') as f:
        pickle.dump(potential_positives,f)



