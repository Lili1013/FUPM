import torch
import numpy as np
from sklearn.cluster import KMeans,SpectralClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity



def k_means_1(num_clusters,embeddings,overlap_users):
    kmeans = KMeans(n_clusters=num_clusters,random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    label_dict = {}
    label_count_dict = Counter(cluster_labels)
    for id,label in enumerate(cluster_labels):
        # centroid = centroids[label]
        label_dict[id] = label
    cluster_overlap_smaples = {}
    for i in range(len(embeddings)):
        cluster_label = cluster_labels[i]
        centroid = centroids[cluster_label]
        if cluster_label not in cluster_overlap_smaples:
            cluster_overlap_smaples[cluster_label] = [[],[],[],[label_count_dict[cluster_label]]]
        if i in overlap_users:
            if list(centroid) not in cluster_overlap_smaples[cluster_label]:
                cluster_overlap_smaples[cluster_label][0].extend(centroid)
            cluster_overlap_smaples[cluster_label][1].append(i)
            sim = cosine_similarity(embeddings[i].reshape(1,-1),centroid.reshape(1,-1))[0][0]
            cluster_overlap_smaples[cluster_label][2].append(sim)
        else:
            if list(centroid) not in cluster_overlap_smaples[cluster_label]:
                cluster_overlap_smaples[cluster_label][0].extend(centroid)
    return label_dict,cluster_overlap_smaples,set(cluster_labels)

def k_means(num_clusters,embeddings,overlap_users):
    kmeans = KMeans(n_clusters=num_clusters,random_state=42)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    label_dict = {}
    label_count_dict = Counter(cluster_labels)
    # for id,label in enumerate(cluster_labels):
    #     # centroid = centroids[label]
    #     label_dict[id] = label
    cluster_overlap_smaples = {}
    for i in range(len(embeddings)):
        cluster_label = cluster_labels[i]
        centroid = centroids[cluster_label]
        cluster_overlap_smaples[i] = [[],[],[]]
        cluster_overlap_smaples[i][0].append(cluster_label)
        cluster_overlap_smaples[i][1].extend(centroid)
        sim = cosine_similarity(embeddings[i].reshape(1, -1), centroid.reshape(1, -1))[0][0]
        cluster_overlap_smaples[i][2].append(sim)
        # if cluster_label not in cluster_overlap_smaples:
        #     cluster_overlap_smaples[cluster_label] = [[],[],[],[label_count_dict[cluster_label]]]
        # if i in overlap_users:
        #     if list(centroid) not in cluster_overlap_smaples[cluster_label]:
        #         cluster_overlap_smaples[cluster_label][0].extend(centroid)
        #     cluster_overlap_smaples[cluster_label][1].append(i)
        #     sim = cosine_similarity(embeddings[i].reshape(1,-1),centroid.reshape(1,-1))[0][0]
        #     cluster_overlap_smaples[cluster_label][2].append(sim)
        # else:
        #     if list(centroid) not in cluster_overlap_smaples[cluster_label]:
        #         cluster_overlap_smaples[cluster_label][0].extend(centroid)
    return cluster_overlap_smaples,set(cluster_labels),centroids

def select_K(max_clusters,embeddings):
    sse = []
    for k in range(2,max_clusters):
        k_means = KMeans(n_clusters=k,random_state=42)
        k_means.fit(embeddings)
        sse.append(silhouette_score(embeddings,k_means.labels_))
    plt.plot(range(2,max_clusters),sse)
    plt.show()


def kmeans_temp(clusters,embeddings):
    k_means = KMeans(n_clusters=clusters, random_state=42)
    k_means.fit(embeddings)

    centroids = k_means.cluster_centers_
    cluster_labels = k_means.labels_
    return centroids,cluster_labels


def draw(embeddings,labels,centers):
    plt.scatter(embeddings[:,0],embeddings[:,1],c=labels,cmap='viridis',edgecolors='k',s=50)
    plt.scatter(centers[:,0],centers[:,1],c='red',marker='X',s=200,label='Cluster Centers')
    plt.savefig('jj.pdf')
    plt.show()

def spectral_cluster(num_clusters,embeddings,overlap_users):
    spectral = SpectralClustering(n_clusters=num_clusters,affinity='nearest_neighbors',random_state=42)
    cluster_labels = spectral.fit_predict(embeddings)
    print('gg')
    centroids = []
    for cluster_label in np.unique(cluster_labels):
        cluster_points = embeddings[cluster_labels==cluster_label]
        cluster_center = np.mean(cluster_points,axis=0)
        centroids.append(cluster_center)
    label_dict = {}
    label_count_dict = Counter(cluster_labels)
    for id, label in enumerate(cluster_labels):
        # centroid = centroids[label]
        label_dict[id] = label
    cluster_overlap_smaples = {}
    for i in range(len(embeddings)):
        cluster_label = cluster_labels[i]
        centroid = centroids[cluster_label]
        if cluster_label not in cluster_overlap_smaples:
            cluster_overlap_smaples[cluster_label] = [[],[],[label_count_dict[cluster_label]]]
        if i in overlap_users:
            if list(centroid) not in cluster_overlap_smaples[cluster_label]:
                cluster_overlap_smaples[cluster_label][0].extend(centroid)
            cluster_overlap_smaples[cluster_label][1].append(i)
        else:
            if list(centroid) not in cluster_overlap_smaples[cluster_label]:
                cluster_overlap_smaples[cluster_label][0].extend(centroid)
    return label_dict,cluster_overlap_smaples,set(cluster_labels)



if __name__ == '__main__':
    # path = '/data/lwang9/CDR_data_process/douban_data_process_FedPCL_MDR/datasets/douban_book/doc_embs/book_user_doc_emb_32.npy'
    # with open(path, 'rb') as f_user:
    #     user_review_emb = np.load(f_user)
    data = np.random.rand(100,2)
    num_clusters = 3
    overlap_users = [i for i in range(5)]
    centroids, cluster_labels = k_means(num_clusters,data,overlap_users)
    print(cluster_labels)

    draw(embeddings=data,labels=cluster_labels,centers=centroids)
    # select_K(max_clusters=num_clusters,embeddings=user_review_emb)
    # overlap_users = [i for i in range(50)]
    # label_dict,cluster_overlap_smaples,cluster_labels = k_means(num_clusters=num_clusters,embeddings=data,overlap_users=overlap_users)
    # spectral_cluster(num_clusters,data,overlap_users)
    print('hh')
