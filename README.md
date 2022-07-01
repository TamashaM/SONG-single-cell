# SONG-single-cell
This Repository Contains the Source Code for the Python Implementation of "Dimensionality reduction for visualizing high-dimensional biological data"
```
1.data_loader.py
```
Contains the code for loading and preprocessing the data. Make sure to download the datasets to the /data/ directory prior to using the script.

```
2.artifical_tree.py 
```
Contains the code used to generate the simulated dataset containing the continuous branches (Figure 2). This code is adapted from the code base provided by the following paper

Moon, K.R., van Dijk, D., Wang, Z. et al. Visualizing structure and transitions in high-dimensional biological data. Nat Biotechnol 37, 1482–1492 (2019). 

```
3.artificial_tree_with_blob.py
```
Contains the code used to generate the simulated dataset containing both continuous branches and discrete clusters (Figure 3).

```
4.visualization_generator.py
```
Contains the code used to generate the visualizations corresponding to real biological data.

```
5.pairwise_experiment.py
```
Contains the code used to generate the high dimensional and low dimensional pairwise distane correlations graphs. (Figure 10)

