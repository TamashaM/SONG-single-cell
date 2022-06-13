import data_loader

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn import preprocessing

from song.song import SONG
from sklearn.decomposition import PCA
import umap
import phate

from scipy.sparse import csr_matrix

import csv
import time


def create_model(model_name, dataset, n_components, random_seed):
    if model_name == "SONG" or model_name == "SONG_PCA":
        model = SONG(n_components=n_components, dispersion_method="UMAP", random_seed=random_seed)
    elif model_name == "PHATE" or model_name == "PHATE_PCA":
        model = phate.PHATE(n_components=n_components, n_jobs=-2, random_state=random_seed)
    elif model_name == "UMAP" or model_name == "UMAP_PCA":
        model = umap.UMAP(n_components=n_components, random_state=random_seed)
    elif model_name == "PCA":
        model = PCA(n_components=n_components, random_state=random_seed)
    else:
        print("unknown model name", model_name)

    return model


def clustering(method, n_clusters, embedding):
    if method == "kmeans":
        clustering = KMeans(n_clusters=n_clusters)
    elif method == "hc":
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        print("Unknown clustering method {}".format(method))

    clustering.fit(embedding)
    labels = clustering.labels_

    return labels


datasets = ["bone_marrow", "epithelial", "samusik_01", "EB", "mouse_cortical", "mouse_cortex_2", "mouse_retinal"]
# datasets = ["mouse_retinal"]

experiment_type="song"
if experiment_type == "song":
    model_names = ["SONG", "SONG_PCA"]
elif experiment_type=="umap":
    model_names = ["UMAP","UMAP_PCA"]
elif experiment_type == "phate":
    model_names = ["PHATE","PHATE_PCA"]


n_components = [2, 3]
clustering_methods = ["kmeans", "hc"]

with open('./results_clustering_final/nmi_{}.csv'.format(experiment_type), 'w') as f, open('./results_clustering_final/ari_{}.csv'.format(experiment_type),
                                                                      'w') as g, open(
        './results_clustering_final/times_{}.csv'.format(experiment_type), 'w') as t:
    writer1 = csv.writer(f)
    writer2 = csv.writer(g)
    writer3 = csv.writer(t)
    for dataset in datasets:
        print("Dataset {}".format(dataset))
        data, true_labels = data_loader.load_data(dataset)
        print("true_labels", true_labels)

        le = preprocessing.LabelEncoder()
        true_labels = le.fit_transform(true_labels)

        print("true_labels encoded", true_labels)
        n_cells = len(data)
        n_clusters = len(set(true_labels))

        for n_component in n_components:
            print("n_components", n_component)
            for model_name in model_names:
                row_f = []
                row_g = []
                row_t = []

                row_f.append(dataset)
                row_f.append(n_component)
                row_f.append(model_name)

                row_g.append(dataset)
                row_g.append(n_component)
                row_g.append(model_name)

                row_t.append(dataset)
                row_t.append(n_component)
                row_t.append(model_name)

                k_means_nmi = []
                k_means_ari = []
                hc_nmi = []
                hc_ari = []

                model_times = []
                print("training the embeddings with {}".format(model_name))
                for i in range(10):
                    print("run {}".format(i))
                    model = create_model(model_name=model_name, dataset=dataset, n_components=n_component,
                                         random_seed=i)
                    if model_name == "UMAP" or model_name == "SONG" or model_name == "PHATE" or model_name == "PCA":
                        start = time.time()
                        if dataset == "samusik_01" or dataset == "mouse_thymus":
                            Y = model.fit_transform(data)
                        else:
                            Y = model.fit_transform(csr_matrix(data))
                        end = time.time()
                        time_spent = end - start
                    else:
                        print(model_name, "transforming with PCA")
                        n = min(len(data), len(data[0]))
                        n_components_pca = min(n, 100)
                        print("n_components for PCA", n_components_pca)
                        transformed_data = PCA(n_components=n_components_pca).fit_transform(data)
                        start = time.time()
                        Y = model.fit_transform(transformed_data)
                        end = time.time()
                        time_spent = end - start
                    model_times.append(time_spent)

                    for clustering_method in clustering_methods:
                        print("evaluating the clustering performance with {}".format(clustering_method))

                        pred_labels = clustering(method=clustering_method, n_clusters=n_clusters, embedding=Y)

                        NMI = normalized_mutual_info_score(pred_labels, true_labels)
                        ARI = adjusted_rand_score(pred_labels, true_labels)
                        print(NMI, ARI)

                        if clustering_method == "kmeans":
                            k_means_nmi.append(NMI)
                            k_means_ari.append(ARI)
                        else:
                            hc_nmi.append(NMI)
                            hc_ari.append(ARI)
                writer1.writerow(row_f + ["kmeans"] + k_means_nmi)
                writer1.writerow(row_f + ["hc"] + hc_nmi)
                writer2.writerow(row_g + ["kmeans"] + k_means_ari)
                writer2.writerow(row_g + ["hc"] + hc_ari)
                writer3.writerow(row_t + model_times)
                f.flush()
                g.flush()
                t.flush()
f.close()
g.close()
t.close()
