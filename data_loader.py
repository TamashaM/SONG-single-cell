import pandas as pd
import scprep
import fcsparser
import numpy as np
import os

BMMC_label_map = {
    1: "Erythrocyte",
    2: "Erythrocyte",
    3: "Erythrocyte",
    4: "Erythrocyte",
    5: "Erythrocyte",
    6: "Erythrocyte",
    7: "Early Erythrocyte",
    8: "Megakaryocyte",
    9: "Early Neutrophil",
    10: "Early Monocyte",
    11: "Dendrytic",
    12: "Early Basophil",
    13: "Basophil",
    14: "Monocyte",
    15: "Monocyte",
    16: "Neutrophil",
    17: "Neutrophil",
    18: "Eosinophil",
    19: "Lymphoid"
}
OCT4_filenames = [
    "4FI_00.fcs",
    "4FI_01.fcs",
    "4FI_02.fcs",
    "4FI_04.fcs",
    "4FI_06.fcs",
    "4FI_08.fcs",
    "4FI_10.fcs",
    "4FI_12.fcs",
    "4FI_14.fcs",
    "4FI_16.fcs",
    "4FI_17.fcs",
    "4FI_18.fcs",
    "4FI_20.fcs",
]


def preprocess_data(data, labels):
    # # remove rare genes
    data = scprep.filter.filter_rare_genes(data, min_cells=10)
    # normalization
    data = scprep.normalize.library_size_normalize(data)
    # transformation
    data = scprep.transform.log(data)

    return data, labels


def load_data(dataset):
    directory_path = "./data/{}/".format(dataset)

    if dataset == "bone_marrow":
        labels = pd.read_csv(os.path.join(directory_path, "MAP.csv"), names=["name", "label"])
        labels = labels.replace({"label": BMMC_label_map})
        data = scprep.io.load_csv(os.path.join(directory_path, "BMMC_myeloid.csv"))
        labels = labels.set_index('name')
        data, labels = preprocess_data(data, labels)
        data = data.values.astype(np.float32)
        labels = labels["label"].tolist()


    elif dataset == "epithelial":
        data = pd.read_csv(directory_path + "GSE92332_atlas_UMIcounts.txt", delimiter="\t").T
        label_data = pd.read_csv(directory_path + "GSE92332_atlas_UMIcounts.txt", delimiter="\t").T
        labels = [x.split("_")[2] for x in label_data.index]
        print(data.head())
        data = data.iloc[:, 1:].values.astype(np.float32)
        print(data)
        data, labels = preprocess_data(data, labels)


    elif dataset == "mouse_thymus":
        meta, data = fcsparser.parse(directory_path + "sample_masscyt.fcs", meta_data_only=False)
        print(data)
        labels = data["CD4"].tolist()
        data = data.iloc[:, 1:].values.astype(np.float32)


    elif dataset == "EB":
        sparse = True
        download_path = "./data/EB/"
        T1 = scprep.io.load_10X(os.path.join(download_path, "scRNAseq", "T0_1A"), sparse=sparse, gene_labels='both')
        T2 = scprep.io.load_10X(os.path.join(download_path, "scRNAseq", "T2_3B"), sparse=sparse, gene_labels='both')
        T3 = scprep.io.load_10X(os.path.join(download_path, "scRNAseq", "T4_5C"), sparse=sparse, gene_labels='both')
        T4 = scprep.io.load_10X(os.path.join(download_path, "scRNAseq", "T6_7D"), sparse=sparse, gene_labels='both')
        T5 = scprep.io.load_10X(os.path.join(download_path, "scRNAseq", "T8_9E"), sparse=sparse, gene_labels='both')

        filtered_batches = []
        for batch in [T1, T2, T3, T4, T5]:
            batch = scprep.filter.filter_library_size(batch, percentile=20, keep_cells='above')
            batch = scprep.filter.filter_library_size(batch, percentile=75, keep_cells='below')
            filtered_batches.append(batch)
        del T1, T2, T3, T4, T5

        EBT_counts, sample_labels = scprep.utils.combine_batches(
            filtered_batches,
            ["Day 00-03", "Day 06-09", "Day 12-15", "Day 18-21", "Day 24-27"],
            append_to_cell_names=True
        )
        del filtered_batches  # removes objects from memory
        EBT_counts = scprep.filter.filter_rare_genes(EBT_counts, min_cells=10)
        EBT_counts = scprep.normalize.library_size_normalize(EBT_counts)
        mito_genes = scprep.select.get_gene_set(EBT_counts,
                                                starts_with="MT-")  # Get all mitochondrial genes. There are 14, FYI.

        EBT_counts, sample_labels = scprep.filter.filter_gene_set_expression(
            EBT_counts, sample_labels, genes=mito_genes,
            percentile=90, keep_cells='below')
        data = np.array(scprep.transform.sqrt(EBT_counts)).astype(np.float32)
        labels = sample_labels

    elif dataset == "cytof":
        OCt4 = [
            pd.read_csv(directory_path + "fcs_gated/" + fname + "_file_internal_comp_export.txt", delimiter="\t",
                        skiprows=1).values
            for fname in OCT4_filenames]
        all_data = np.vstack(f for f in OCt4)
        data = all_data[:, 3:-2]
        times = [int(s[4:6]) for s in OCT4_filenames]
        day = np.hstack(np.ones(len(n)) * times[i] for i, n in enumerate(OCt4))
        labels = day

    elif dataset == "mouse_retinal":
        fname = "output.txt"
        fname_labels = "clust_retinal_bipolar.txt"
        data = pd.read_csv(directory_path + fname, delimiter="\t", index_col=0, header=None).T
        labels = pd.read_csv(directory_path + fname_labels, delimiter="\t")
        labels = [x.split(" ")[0] for x in labels["CLUSTER"].tolist()]
        data = data.iloc[:, 1:].astype(float)

    elif dataset == "mouse_cortical":
        fname = "zeisel_counts.csv"
        fname_label = "zeisel_coldata.csv"
        data = pd.read_csv(directory_path + fname, index_col=0).T
        meta = pd.read_csv(directory_path + fname_label, index_col=0)
        print(meta.head())

        labels = meta["cell_type1"]
        data, labels = preprocess_data(data, labels)
        data = data.values.astype(np.float32)

    elif dataset == "samusik_01":
        data = pd.read_csv(directory_path + "exp.csv")
        labels = pd.read_csv(directory_path + "populations.csv")
        label_annot = pd.read_csv(directory_path + "/population_names_Samusik_and_gating.csv")
        label_annot = label_annot.append({"population": 0, "name": "Ungated"}, ignore_index=True)
        print(label_annot)
        labels["x_label"] = labels["x"].map(label_annot.set_index("population")["name"])

        data["labels"] = labels["x_label"].tolist()
        # data_filtered = data.dropna()
        labels = data["labels"].tolist()
        print(labels)
        data = data.iloc[:, 1:-1].values.astype(np.float32)

    elif dataset == "samusik_all":
        data = pd.read_csv(directory_path + "exp_all.csv")
        labels = pd.read_csv(directory_path + "populations.csv")
        data["labels"] = labels["x"].tolist()
        data_filtered = data.dropna()
        data = data_filtered.iloc[:, 1:-1].astype(float)
        labels = data_filtered["labels"].tolist()

    elif dataset == "HSMM":
        data = pd.read_csv(directory_path + "outfile.csv", index_col=0, header=None).T
        labels = pd.read_csv(directory_path + "time_points.csv")["V1"].tolist()
        data = data.iloc[:, 1:].astype(float)

    elif dataset == "mouse_cortex_2":
        data = pd.read_csv(directory_path + "outfile.csv", index_col=0, header=None).T
        labels = pd.read_csv(directory_path + "time_points.csv")["V1"].tolist()
        data = data.iloc[:, 1:].values.astype(float)

    elif dataset == "keyboard":
        data=np.load("./data/keyboard/array.npy")
        labels=None

    elif dataset == "soils":
        data=np.load("./data/soils/array.npy")
        labels=None

    elif dataset == "hmp":
        data=np.load("./data/hmp/array.npy")
        labels=None

    elif dataset == "pop_gen":
        data=np.load("./data/pop_gen/array.npy")
        labels=None

    return data, labels
