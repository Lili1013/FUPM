import pandas as pd
import numpy as np
from loguru import logger

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords  # import stopwords corpus
import string  # delete various punctuation
import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('punkt')

def tokenize_text(text):
    # print(text)
    tokens = word_tokenize(str(text).lower())
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    return tokens
stop_words = set(stopwords.words('english'))

def concat_reviews(df,df_map,column_name):
    id_lists = []
    review_text_lists =[]
    ids = list(df[column_name].unique())
    for index,row in df_map.iterrows():
        id = row[column_name]
        id_lists.append(id)
        if id in ids:
            concat_review = ';'.join(df[df[column_name]==id]['review_texts'])
        else:
            concat_review = '<empty>'
        review_text_lists.append(concat_review)
    df_concat_review = pd.DataFrame({
        column_name:id_lists,
        'review_texts':review_text_lists
    })
    return df_concat_review

def generate_diff_ratio_train_data(datasets,domain,ratio_datasets):
    to_datasets = ratio_datasets
    df = pd.read_csv(f'../../datasets/{datasets}/{domain}/train.csv')
    np.random.seed(42)
    ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ratio_dfs = []
    root_path = f'../../datasets/{to_datasets}/{datasets}_'
    paths = [root_path + '']
    for ratio in ratios:
        print(ratio)
        grouped = df.groupby('userID')
        sample_df = grouped.apply(lambda x: x.sample(frac=ratio)).reset_index(drop=True)
        # sample_df = id_map(sample_df)
        sample_df.to_csv(root_path + str(ratio) + '/' + domain + '/train.csv', index_label=False)

def process_review_data(datasets,domain,ratio_datasets,ratio,user_map_df,item_map_df):
    review_path = f'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/{datasets}/{domain}/{domain}_reviews_orig.csv'
    df_review = pd.read_csv(review_path)
    train_path = f'../../datasets/{ratio_datasets}/{datasets}_{ratio}/{domain}/train.csv'
    test_path = f'../../datasets/{datasets}/{domain}/test.csv'
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    df = pd.concat([train_df,test_df],axis = 0)
    df['review_texts'] = ''
    df['user_id'] = 0
    df['item_id'] = 0
    for index,row in df.iterrows():
        userID = row['userID']
        itemID = row['itemID']
        user_id = user_map_df[user_map_df['userID']==userID]['user_id'].iloc[0]
        item_id = item_map_df[item_map_df['itemID'] == itemID]['item_id'].iloc[0]
        review = df_review[(df_review['user_id']==user_id)&(df_review['item_id']==item_id)]['review_text'].iloc[0]
        df.loc[(df['userID'] == userID) & (df['itemID'] == itemID), 'user_id'] = user_id
        df.loc[(df['userID'] == userID) & (df['itemID'] == itemID), 'item_id'] = item_id
        df.loc[(df['userID'] == userID) & (df['itemID'] == itemID), 'review_texts'] = review
    return df

def doc2vec_model(dataset,domain,ratio_datasets,name,column_name,emb_size,df,ratio):
    to_path = f'../../datasets/{ratio_datasets}/{dataset}_{ratio}/{domain}/'
    # rev_path = f'{path}{dataset}_{name}_reviews.csv'
    model_save_path = f'{to_path}{domain}_{name}_doc_model_{emb_size}.model'
    # emb_save_path = f'{path}doc_embs/{dataset}_{name}_doc_emb_64.pickle'
    npy_emb_save_path = f'{to_path}{domain}_{name}_doc_emb_{emb_size}.npy'
    logger.info('read file')
    # tokennize and delete stopwords
    logger.info('download stopwords')

    # stop_words = read_stop_words()
    # generate TaggedDocument object for each user
    tagged_data = []
    logger.info('process documents')
    for index, row in df.iterrows():
        documents = [TaggedDocument(words=tokenize_text(row['review_texts']), tags=[row[column_name]])]
        tagged_data.extend(documents)

    # train Doc2Vec model
    logger.info('train doc2vec model')
    model = Doc2Vec(vector_size=emb_size, window=5, min_count=1, workers=4, epochs=50)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(model_save_path)
    # # # obtain each user's Doc2Vec embedding
    logger.info('obtain user embeddings')
    user_embeddings = {}
    for index, row in df.iterrows():
        user_id = row[column_name]
        user_embedding = model.dv[user_id]
        user_embeddings[user_id] = user_embedding
    all_vectors = []
    # # with open(emb_save_path, 'rb') as f:
    # #     text_feat = pickle.load(f)
    for key, value in user_embeddings.items():
        all_vectors.append(value)
    vectors_array = np.array(all_vectors)
    np.save(npy_emb_save_path, vectors_array)


if __name__ == '__main__':
    datasets = 'amazon_phone_sport'
    domain = 'phone'
    ratio_datasets = 'amazon_phone_sport_diff_inter_ratio'
    ratio = '0.4'
    # #first step: generate the train data with different ratio of user interactions
    # generate_diff_ratio_train_data(datasets=datasets,domain=domain,ratio_datasets=ratio_datasets)
    # #second step: process review data
    user_map_path = f'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/{datasets}/{domain}/user_id_map.csv'
    item_map_path = f'/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/{datasets}/{domain}/item_id_map.csv'
    user_map_df = pd.read_csv(user_map_path)
    item_map_df = pd.read_csv(item_map_path)
    logger.info('process review data')
    df = process_review_data(datasets=datasets,domain=domain,ratio_datasets=ratio_datasets,ratio=ratio,
                             user_map_df=user_map_df,item_map_df=item_map_df)
    df.to_csv(f'../../datasets/{ratio_datasets}/{datasets}_{ratio}/{domain}/review_data.csv',index=False)
    # #third step: generate doc embedding
    # column_name = 'item_id'
    # df = pd.read_csv(f'../../datasets/{ratio_datasets}/{datasets}_{ratio}/{domain}/review_data.csv')
    # df['review_texts'] = df['review_texts'].astype(str)
    # logger.info('concat reviews')
    # df_concat_review = concat_reviews(df=df,df_map=item_map_df,column_name=column_name)
    # logger.info('generate doc emb')
    # doc2vec_model(dataset=datasets,domain=domain,ratio_datasets=ratio_datasets,name=column_name.split('_')[0],column_name=column_name,
    #               emb_size=256,df=df_concat_review,ratio = ratio)

