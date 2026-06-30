import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans,SpectralClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch
from loguru import logger

def select_K_1(max_clusters,embeddings):
    sse = []
    for k in range(5,max_clusters):
        k_means = KMeans(n_clusters=k,random_state=42)
        k_means.fit(embeddings)
        sse.append(silhouette_score(embeddings,k_means.labels_))
    plt.plot(range(5,max_clusters),sse)
    plt.show()

def select_K(max_clusters,embeddings):
    sse = []
    for k in range(2,max_clusters):
        logger.info(k)
        k_means = MiniBatchKMeans(n_clusters=k,random_state=42,batch_size=100)
        k_means.fit(embeddings)
        sse.append(silhouette_score(embeddings,k_means.labels_))
    plt.plot(range(2,max_clusters),sse)
    plt.show()

def k_means(num_clusters,embeddings):
    kmeans = KMeans(n_clusters=num_clusters,random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    label_dict = {}
    label_count_dict = Counter(cluster_labels)
    # for id,label in enumerate(cluster_labels):
    #     # centroid = centroids[label]
    #     label_dict[id] = label
    # cluster_overlap_smaples = {}
    centroid_samples = {}
    # for i in range(len(embeddings)):
    #     cluster_label = cluster_labels[i]
    #     centroid = centroids[cluster_label]
    #     cluster_overlap_smaples[i] = [[],[],[]]
    #     cluster_overlap_smaples[i][0].append(cluster_label)
    #     cluster_overlap_smaples[i][1].extend(centroid)
    #     sim = cosine_similarity(embeddings[i].reshape(1, -1), centroid.reshape(1, -1))[0][0]
    #     cluster_overlap_smaples[i][2].append(sim)
    for i in range(kmeans.n_clusters):
        centroid_samples[i] = list(np.where(cluster_labels==i)[0])
    centroid_dict = {i:list(centroids[i]) for i in range(len(centroids))}
    return centroid_dict,centroid_samples

def update_cat(category_dict,df):
    for key,value in category_dict.items():
        for each_value in value:
            df.loc[df['userID'] == int(each_value), 'cat'] = int(key)
    return df

if __name__ == '__main__':
    with open('/data/lwang9/CDR_data_process/data_process_FedPCL_MDR_imp_common_user/datasets/book_music/music/music_item_doc_emb_sbert.npy', 'rb') as f_user:
        sport_item_review_emb = torch.from_numpy(np.load(f_user))
    # with open('../../datasets/phone_sport/phone/phone_item_doc_emb_32.npy', 'rb') as f_item:
    #     phone_item_review_emb = torch.from_numpy(np.load(f_item))
    select_K(20,sport_item_review_emb)
    # df_phone = pd.read_csv('/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/phone/phone_inter.csv')
    # df_sport = pd.read_csv('/data/lwang9/CDR_data_process/amazon_data_process_P2M2_CDR/datasets/phone_sport/sport/sport_inter.csv')
    # phone_centroids,phone_centroid_samples = k_means(10,phone_item_review_emb)
    # with open('../../datasets/phone_sport/phone/phone_cluster_samples.pkl','wb') as f:
    #     pickle.dump(phone_centroid_samples,f)
    #
    # with open('../../datasets/phone_sport/phone/phone_cluster_samples.pkl','rb') as f:
    #     data = pickle.load(f)
    # print('hh')

    # sport_centroids,sport_centroid_samples = k_means(20, sport_user_review_emb)
    # df_phone['cat'] = 0
    # df_sport['cat'] = 0
    # df_phone = update_cat(phone_centroid_samples,df_phone)
    # print('hh')
    # df_sport = update_cat(sport_centroid_samples,df_sport)