import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances
import seaborn as sns
import data_loader
from scipy.sparse import csr_matrix

np.random.seed(4)


models = ["SONG", "UMAP", "PHATE"]
dataset_name = {
    "bone_marrow": "B) Paul et al.(2015)",
    "epithelial": "D) Haber et al.(2017)",
    "samusik_01": "A) Samusik et al.(2016)",
    "EB": "E) Moon et al.(2019)",
    "mouse_cortical": "C) Zeisel et al.(2015)",
    "mouse_cortex_2": "F) Yuzwa et al.(2017)",
    "mouse_thymus": "G) Setty et al.(2016)",
    "mouse_retinal": "H) Shekhar et al.(2016)",
    "keyboard": "I) Fierer et al. (2010)",
    "soils": "J) Lauber et al. (2009)",
    "hmp": "K) Huttenhower et al. (2012)",
    "pop_gen": "L) Auton et al. (2015)"
}

datasets = ["bone_marrow","epithelial"]
# datasets = ["EB", "mouse_cortex_2","mouse_thymus", "mouse_retinal"]

fig, axes = plt.subplots(len(models) + 1, len(datasets), figsize=(10, 8))
for j, dataset in enumerate(datasets):
    print("processing {}".format(dataset))
    data, labels = data_loader.load_data(dataset)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data=np.array(data))

    if len(data.index) > 1000:
        sample_size = 1000
    else:
        sample_size = len(data.index)
    data_sample = data.reset_index(drop=True).sample(sample_size)
    data_pd = pairwise_distances(data_sample)
    data_df = pd.DataFrame({"original": np.array(data_pd).flatten()})

    for i, model_name in enumerate(models):
        print("at model {}".format(model_name))

        embedding = pd.read_pickle("./results_extension/{}/{}.pkl".format(dataset, model_name))
        embedding_sample = embedding.iloc[data_sample.index]
        embedding_pd = pairwise_distances(embedding_sample.iloc[:, :-1])  # leave label column
        embedding_df = pd.DataFrame(np.array(embedding_pd).flatten())

        data_df["embedding"] = embedding_df.iloc[:, -1]

        max = data_df.iloc[:, 0].max()
        min = data_df.iloc[:, 0].min()
        gap = (max - min) / 50

        bin_labels = [i * gap for i in range(50)]
        bins = pd.cut(data_df["original"], bin_labels)
        grouped_embedding_df = data_df.groupby(bins)

        data_array = []
        found = False
        for name, grouped in grouped_embedding_df:
            data_array.append(grouped["embedding"].tolist())
        pearson = pearsonr(np.array(data_pd).flatten(), np.array(embedding_pd).flatten())
        g = sns.boxplot(data=data_array, ax=axes[i, j], linewidth=1, fliersize=0, color='white')
        g.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        g.set_title("r = {}".format(round(pearson[0], 2)))
        if j == 0:
            g.set_ylabel(model_name, fontweight='bold')
        if i == len(models) - 1:
            g.set_xlabel(dataset_name[dataset], fontweight='bold')

    hist_plot = sns.histplot(data_df.iloc[:, 0], bins=50, ax=axes[i + 1, j], edgecolor='black',
                             legend=False)
    hist_plot.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=True,
        labelleft=False)
    hist_plot.set_xlabel("original distance")
image_path = "./results_extension/min_dist{}".format("all")
plt.savefig(image_path + ".png", bbox_inches='tight')
plt.savefig(image_path + ".svg", bbox_inches='tight')
plt.show()
