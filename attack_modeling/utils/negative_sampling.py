import pandas as pd
import pickle
from loguru import logger

# dataset = 'book'
# domain = 'douban_book_music'
dataset = 'sport'
domain = 'amazon_phone_sport'
df = pd.read_csv(f'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/{domain}/{dataset}/{dataset}_inter.csv')
# df = pd.read_csv(f'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/'
#                  f'{domain}/{dataset}/{dataset}_review_data.csv')
user_ids = list(df['userID'].unique())
logger.info(len(user_ids))
item_ids = set(df['itemID'].unique())
with open(f'../../../datasets/{domain}/{dataset}/user_pos_items_doc_emb_256_0.3.pkl', 'rb') as f:
    potential_pos_dict = pickle.load(f)
user_inter_dict = df.groupby('userID')['itemID'].apply(list).to_dict()
user_dislike_dict = dict()
for key,value in user_inter_dict.items():
    logger.info(key)
    potential_items = []
    for each_item in value:
        potential_items.extend(potential_pos_dict[(key,each_item)])
    potential_items.extend(value)
    potential_items = set(potential_items)
    dislike_items = item_ids-potential_items
    dislike_items = list(dislike_items)
    if len(dislike_items) == 0:
        dislike_items = list(item_ids-set(value))
    user_dislike_dict[key] = dislike_items

with open(f'../../../datasets/{domain}/{dataset}/user_dislike_items_0.3.pkl','wb') as f:
    pickle.dump(user_dislike_dict,f)




# with open(f'../../../datasets/amazon_phone_sport/phone/user_dislike_items.pkl', 'rb') as f:
#     dislike_dict = pickle.load(f)
# with open(f'../../../datasets/douban_book_music/book/user_dislike_items.pkl', 'rb') as f:
#     dislike_dict_1 = pickle.load(f)
# print('gg')
# num = 0
# for key,value in dislike_dict.items():
#     if len(value) == 0:
#         num += 1
# print(num)

