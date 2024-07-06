import pandas as pd
import pickle
import random

if __name__ == '__main__':
    df = pd.read_csv(
        '/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/phone/phone_inter.csv')
    user_inter_counts = df.groupby('userID').size().to_dict()
    with open('../../datasets/phone_sport/phone/user_potential_items.pkl', 'rb') as f:
        data = pickle.load(f)
    null_num = []
    user_ids = []
    potential_item_ids = []
    for x, value in data.items():
        if user_inter_counts[x] >= 10:
            continue
        if len(value) > 5:
            value = random.sample(value, 5)
        for each_value in value:
            user_ids.append(x)
            potential_item_ids.append(each_value)
    potential_df = pd.DataFrame({
        'userID': user_ids,
        'itemID': potential_item_ids,
        'rating': 1
    })
    df_all = pd.concat([df[['userID', 'itemID', 'rating']], potential_df], ignore_index=True)
    df_all = df_all.sort_values(by='userID')
    df_all.to_csv('../../datasets/phone_sport/phone/phone_inter_new.csv.csv',index=False)


