import pickle
import pandas as pd
from loguru import logger

dataset = 'douban_book_music'
domain = 'music'
# df = pd.read_csv(
#         f'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/{dataset}/{domain}/{domain}_inter.csv')
# users = df['userID'].unique()
# all_items = set(df['itemID'].unique())

# num=0
# with open(f'../../datasets/{dataset}/{domain}/user_pos_items_doc_emb_sbert_0.5.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print('gg')
# for ley,value in data.items():
#     if len(value) == 0:
#         num+=1
# print(num)
# # 遍历每个用户
# num = 0
# for user in users:
#     interacted_items = df[df['userID'] == user]['itemID'].unique()
#     non_interacted_items = list(all_items - set(interacted_items))
#     if len(non_interacted_items)!=len(data[user]):
#         num += 1
#         logger.info(f'non inter items:{len(non_interacted_items)}, neg sampling items:{len(data[user])}')
# logger.info(num)

df = pd.read_csv('/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/douban_book_music/book/book_review_data.csv')
user_inter_dict = df.groupby('userID')['itemID'].apply(list).to_dict()
with open('../../datasets/douban_book_music/music/user_neg_items_doc_emb_sbert_0.5.pkl', 'rb') as f:
    potential_pos_dict = pickle.load(f)
users = df['userID'].unique()
all_items = set(df['itemID'].unique())
user_potential_pos_items_dict = {}
for user_id in users:
    user_potential_pos_items_dict[user_id] = set([item for key, items in potential_pos_dict.items() if key[0] == user_id for item in items])
    print('hh')
for user_id in users:
    user_inter_items = set(user_inter_dict[user_id])
    potential_pos_items = user_potential_pos_items_dict.get(user_id, set())
    non_inter_items = list(all_items - user_inter_items - potential_pos_items)
    print('hh')






