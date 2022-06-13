import pandas as pd
from song.song import SONG
import umap
import os

import phate
import matplotlib.pyplot as plt
import pickle5 as pickle
from sklearn.decomposition import PCA
import data_loader
from pathlib import Path
from scipy.sparse import csr_matrix
from matplotlib.font_manager import FontProperties

import numpy as np

#add the datasets as a python list
datasets = ["bone_marrow"]

model_names = ["SONG", "PHATE", "UMAP"]

for dataset in datasets:
    Path("./results_extension/{}".format(dataset)).mkdir(parents=True, exist_ok=True)
    for ax_i, model_name in enumerate(model_names):
        if model_name == "SONG":
            if dataset == "samusik_01":
                model = SONG(dispersion_method="UMAP", min_dist=0.2)
            else:
                model = SONG(dispersion_method="UMAP")
        elif model_name == "PHATE":
            model = phate.PHATE(n_jobs=-2)
        elif model_name == "UMAP":
            if dataset == "samusik_all" or dataset == "samusik_01":
                print("training with changed params")
                model = umap.UMAP(n_neighbors=15, min_dist=0.2, random_state=123)
            else:
                model = umap.UMAP()

        x_name = model_name + "1"
        y_name = model_name + "2"
        pickle_path = "./results_extension/{}/{}.pkl".format(dataset, model_name)
        image_path = "./results_extension/{}/combined_grid_changed".format(dataset)
        if not os.path.exists(pickle_path):
            data, labels = data_loader.load_data(dataset)
            print("pickle does not exist")
            if model_name == "UMAP":
                if dataset == "samusik_all" or dataset == "samusik_01" or dataset == "mouse_thymus" or dataset == "cytof":
                    Y = model.fit_transform(data)
                else:
                    Y = model.fit_transform(PCA(n_components=100).fit_transform(data))

            else:
                if dataset == "samusik_01" or dataset == "cytof" or dataset == "mouse_thymus":
                    Y = model.fit_transform(data)
                else:
                    Y = model.fit_transform(csr_matrix(data))
            df = pd.DataFrame({x_name: Y.T[0], y_name: Y.T[1], "labels": labels})
            df.to_pickle(pickle_path)

        else:
            print("pickle already exists")
            if dataset == "mouse_retinal" or dataset == "samusik_all":
                df = pickle.load(open(pickle_path, "rb"))
            else:
                df = pd.read_pickle(pickle_path)

        def get_cmap(n, name='Spectral'):
            '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
            RGB color; the keyword argument name must be a standard mpl colormap name.'''
            return plt.cm.get_cmap(name, n)


        label_col_name = df.columns[-1]
        N = len(df[label_col_name].unique())
        cmap = list(get_cmap(20, name="tab20").colors) + list(get_cmap(20, "tab20b").colors)
        if dataset == "samusik_01":
            label_names = list(df[label_col_name].unique())
            label_names.remove("Ungated")
            label_names.insert(0, "Ungated")
            df[label_col_name] = pd.Categorical(df[label_col_name], label_names)
            df = df.sort_values(label_col_name)

        # elif dataset == "mouse_thymus":
        #     df[label_col_name] = df[label_col_name].apply(str)
        if ax_i == 0:
            plt.subplot(1, 2, 1, )
        elif ax_i == 1:
            plt.subplot(2, 2, 2)
        elif ax_i == 2:
            plt.subplot(2, 2, 4)

        if dataset == "mouse_thymus":
            im = plt.scatter(x=df[x_name], y=df[y_name],
                             c=df[label_col_name],
                             cmap="Spectral",
                             s=0.5)
        else:
            grouped_df_result = df.groupby(by=label_col_name)
            for i, (name, group) in enumerate(grouped_df_result):
                name = name.replace("pos", "+")
                name = name.replace("Intermediate Monocytes", "Intermediate \nMonocytes")
                name = name.replace("Non-Classical Monocytes", "Non-Classical \nMonocytes")
                name = name.replace("B-cell fractions A–C (pro-B cells)", "pro-B cells")

                color = cmap[i]
                if name == "Ungated":
                    color = "wheat"
                plt.scatter(x=group[x_name], y=group[y_name],
                            color=color,
                            label=name,
                            s=0.5)
                if dataset != "cytof":
                    plt.annotate(name,
                                 group.iloc[:, 0:2].median(),
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 color="black",
                                 size=4,
                                 )
        plt.xticks([])
        plt.yticks([])
        if dataset == "cytof":
            plt.legend()
        plt.title(model_name)
        # axes[ax_i].set_xlabel(x_name)
        # axes[ax_i].set_ylabel(y_name)


    fontP = FontProperties()
    fontP.set_size('x-small')
    # handles, labels = axes[-2].get_legend_handles_labels()
    # last_sub.legend(handles, labels, loc='center',prop=fontP,markerscale=4,ncol=2)
    # for ax in axes.flatten():
    #     ax.set_alpha(0.5)

    plt.savefig(image_path + ".png", bbox_inches='tight', dpi=1000)
    plt.savefig(image_path + ".svg", bbox_inches='tight')
    plt.show()
