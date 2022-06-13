from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from song.song import SONG
import phate
import numpy as np
import umap
import os


model_names = ["SONG", "PHATE", "UMAP"]
fig, axes = plt.subplots( 3,4)
for j, std in enumerate([5, 10, 15, 20]):
    print("processing {}".format(std))

    tree, branches = phate.tree.gen_dla(n_dim=200, n_branch=10, branch_length=300,
                                        rand_multiplier=2, seed=37, sigma=5)

    X, T = make_blobs(n_samples=10000, centers=10, cluster_std=std, random_state=37, n_features=200)
    T = [(i + 10) for i in T]
    data = np.concatenate((tree, X), axis=0)
    labels = np.concatenate((branches, T), axis=0)

    for k, model_name in enumerate(model_names):
        numpy_path = "./results_final/simulation/{}std{}cp.npy".format(model_name, std)
        print("at model {}".format(model_name))

        if not os.path.exists(numpy_path):

            if model_name == "SONG":
                model = SONG(dispersion_method="UMAP", min_dist=0.5)
            elif model_name == "UMAP":
                model = umap.UMAP(min_dist=0.5)
            elif model_name == "PHATE":
                model = phate.PHATE(gamma=0,t=120)

            Y = model.fit_transform(data.astype(np.float32))
            np.save(numpy_path, Y)

        else:
            Y = np.load(numpy_path)

        axes[k, j].scatter(Y.T[0], Y.T[1], c=labels, cmap=plt.cm.tab20, s=2)
        if k==0:
            if j==0:
                axes[k, j].title.set_text("SD = {}".format(std))
            else:
                axes[k, j].title.set_text(std)
        if j==0:
            axes[k, j].set_ylabel(model_name)

for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
image_path = "./results_extension/simulation/{}".format("allcp")
plt.savefig(image_path + ".png")
plt.savefig(image_path + ".svg")
plt.show()
