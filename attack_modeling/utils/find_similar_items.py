import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pickle
from loguru import logger

dataset = 'amazon_phone_sport'
domain = 'sport'
with open(
        f'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/{dataset}/{domain}/{domain}_item_doc_emb_256.npy',
        'rb') as f_user:
    phone_item_review_emb = torch.from_numpy(np.load(f_user))
    # aa = np.load(f_user)
logger.info('construct sim matrix')
phone_item_review_emb = phone_item_review_emb.numpy()
sparse_item_rev_emb = csr_matrix(phone_item_review_emb)
sim_matrix = cosine_similarity(sparse_item_rev_emb)
similar_items_dict = {}
logger.info('start save')
for idx, row in enumerate(sim_matrix):
    similar_indices = row.argsort()[::-1][1:]
    similar_values = row[similar_indices]
    filter_sim_indices = [i for i,val in zip(similar_indices,similar_values) if val > 0.5]
    similar_items_dict[idx] = filter_sim_indices
with open(f'../../datasets/{dataset}/{domain}/sim_items_0.5.pkl','wb') as f:
    pickle.dump(similar_items_dict,f)
