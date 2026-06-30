import pickle
from loguru import logger

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords  # import stopwords corpus
import string  # delete various punctuation
import numpy as np
import gzip


# download stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')


def tokenize_text(text):
    # print(text)
    tokens = word_tokenize(str(text).lower())
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    return tokens

# def read_stop_words():
#     with open('datasets/stopwords-en.txt','r') as file:
#         file_content = file.read().split('\n')
#     return set(file_content)

# stop_words = read_stop_words()
stop_words = set(stopwords.words('english'))

def concat_reviews(source_path,column_name):
    df = pd.read_csv(source_path)
    df['review_text'].fillna(' ', inplace=True)
    df_lists = []
    # print('start group by')
    logger.info('start group by')
    for x in df.groupby(by=column_name):
        each_df = pd.DataFrame({
            column_name: [x[0]],
            'review_texts': [';'.join(x[1]['review_text'])]
        })
        df_lists.append(each_df)

    df = pd.concat(df_lists, axis=0)
    # print('start store')
    # logger.info('start store')
    # df.to_csv(to_path, index=False)
    return df

def select_reviews(source_path,to_path,items,users):
    g = gzip.open(source_path, 'r')
    review_list = []
    i = 0
    for line in g:
        d = eval(line, {"true": True, "false": False, "null": None})
        if (d['asin'] in items) and (d['reviewerID'] in users):
            if i % 10000 == 0:
                logger.info(i)
            i+=1
            review_list.append([d['reviewerID'], d['asin'],d['reviewText']])
    df = pd.DataFrame(review_list, columns=['user_id', 'item_id','review_text'])  # 转换为dataframe
    df.to_csv(to_path, index=False)
if __name__ == '__main__':
    # df = pd.read_csv('../../datasets/phone_sport/sport/sport_inter_new.csv')
    # user_ids = list(df['userID'].unique())
    # item_ids = list(df['itemID'].unique())
    # user_id_map = pd.read_csv('/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/sport/user_id_map.csv')
    # orig_user_ids = list(user_id_map[user_id_map['userID'].isin(user_ids)]['user_id'].unique())
    # item_id_map = pd.read_csv(
    #     '/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/sport/item_id_map.csv')
    # orig_item_ids = list(item_id_map[item_id_map['itemID'].isin(item_ids)]['item_id'].unique())
    # select_reviews(source_path='/data/lwang9/datasets/amazon/review_texts/reviews_Sports_and_Outdoors.json.gz',
    #                to_path='../../datasets/phone_sport/sport/sport_rev_new.csv',items=orig_item_ids,users=orig_user_ids)


    df = pd.read_csv('../../datasets/phone_sport/phone/phone_rev_new.csv')
    print(len(df))
    df_1 = pd.read_csv('../../datasets/phone_sport/phone/phone_inter_new.csv')
    print(len(df_1))
    # read file
    # # rev_path = '/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/phone/phone_item_reviews.csv'
    # df = concat_reviews(source_path='/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/phone/phone_reviews_orig.csv',column_name='item_id')
    # # rev_path = '../sparsity_datasets/phone_sport/sport/sport_item_reviews.csv'
    # dataset = 'phone'
    # name = 'item'
    # column_name = 'item_id'
    # to_path = f'../datasets/phone_sport/{dataset}/'
    # # rev_path = f'{path}{dataset}_{name}_reviews.csv'
    # model_save_path = f'{to_path}{dataset}_{name}_doc_model_new_32.model'
    # # emb_save_path = f'{path}doc_embs/{dataset}_{name}_doc_emb_64.pickle'
    # npy_emb_save_path = f'{to_path}{dataset}_{name}_doc_emb_new_32.npy'
    # logger.info('read file')
    # # df = pd.read_csv(rev_path)
    # # tokennize and delete stopwords
    # logger.info('download stopwords')
    #
    # # stop_words = read_stop_words()
    # # generate TaggedDocument object for each user
    # tagged_data = []
    # logger.info('process documents')
    # for index, row in df.iterrows():
    #     documents = [TaggedDocument(words=tokenize_text(row['review_texts']), tags=[row[column_name]])]
    #     tagged_data.extend(documents)
    #
    # # train Doc2Vec model
    # logger.info('train doc2vec model')
    # model = Doc2Vec(vector_size=32, window=5, min_count=1, workers=4, epochs=50)
    # model.build_vocab(tagged_data)
    # model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    # model.save(model_save_path)
    # # # obtain each user's Doc2Vec embedding
    # logger.info('obtain user embeddings')
    # user_embeddings = {}
    # for index, row in df.iterrows():
    #     user_id = row[column_name]
    #     user_embedding = model.dv[user_id]
    #     user_embeddings[user_id] = user_embedding
    # #
    # # # save user_embeddings
    # # logger.info('save user embeddings')
    # # with open(emb_save_path,'wb') as file:
    # #     pickle.dump(user_embeddings,file)
    #
    # all_vectors = []
    # # with open(emb_save_path, 'rb') as f:
    # #     text_feat = pickle.load(f)
    # for key, value in user_embeddings.items():
    #     all_vectors.append(value)
    # vectors_array = np.array(all_vectors)
    # np.save(npy_emb_save_path, vectors_array)
