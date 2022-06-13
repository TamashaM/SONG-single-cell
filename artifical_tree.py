from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from song.song import SONG
import phate
import numpy as np
import umap

model_names = ["SONG", "PHATE", "UMAP"]
fig, axes = plt.subplots( 1,3,figsize=(12,3))


tree, branches = phate.tree.gen_dla(n_dim=200, n_branch=10, branch_length=300,
                                    rand_multiplier=2, seed=37, sigma=5)
for k, model_name in enumerate(model_names):
    # numpy_path = "./results_final/simulation/{}std{}.npy".format(model_name, std)
    print("at model {}".format(model_name))



    if model_name == "SONG":
        model = SONG(dispersion_method="UMAP", min_dist=0.9)
    elif model_name == "UMAP":
        model = umap.UMAP(min_dist=0.7,negative_sample_rate=2)
    elif model_name == "PHATE":
        model = phate.PHATE(gamma=0,t=120)

    Y = model.fit_transform(tree.astype(np.float32))


    axes[k].scatter(Y.T[0], Y.T[1], c=branches, cmap=plt.cm.tab20, s=2)
    axes[k].title.set_text(model_name)

for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
image_path = "./results_extension/simulation/artificial_tree{}".format("all")
plt.savefig(image_path + ".png")
plt.savefig(image_path + ".svg")
plt.show()
