import os
import numpy as np
from Bio import SeqIO
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def import_epigenetic_dataset(files_path, cell_line, perc=1.0):
    if cell_line not in ["GM12878", "HelaS3", "HepG2", "K562"]:
        raise ValueError("Illegal cell line.")

    epigenetic_data_path = "{}/{}_200bp_Data.txt".format(files_path, cell_line)
    if not os.path.exists(epigenetic_data_path):
        raise FileNotFoundError("file {} not found".format(epigenetic_data_path))

    labels_path = "{}/{}_200bp_Classes.txt".format(files_path, cell_line)
    if not os.path.exists(labels_path):
        raise FileNotFoundError("file {} not found".format(labels_path))

    logging.debug("Importing epigenetic data of {} cell line".format(cell_line))
    epigenetic_data = np.loadtxt(epigenetic_data_path)

    with open(labels_path, "r") as f:
        labels = np.array([line.strip() for line in f.readlines()])

    X, y = subsample_data(epigenetic_data, labels, perc=perc)

    return X, y


def import_sequence_dataset(files_path, cell_line, perc=1.0):
    if cell_line not in ["GM12878", "HelaS3", "HepG2", "K562"]:
        raise ValueError("Illegal cell line.")

    seqeences_path = "{}/{}.fa".format(files_path, cell_line)
    if not os.path.exists(seqeences_path):
        raise FileNotFoundError("file {} not found".format(seqeences_path))

    labels_path = "{}/{}_200bp_Classes.txt".format(files_path, cell_line)
    if not os.path.exists(labels_path):
        raise FileNotFoundError("file {} not found".format(labels_path))

    logging.debug("Importing sequence data of {} cell line".format(cell_line))
    ltrdict = {'a': [1, 0, 0, 0, 0], 'c': [0, 1, 0, 0, 0],
               'g': [0, 0, 1, 0, 0], 't': [0, 0, 0, 1, 0],
               'n': [0, 0, 0, 0, 1]}
    with open(seqeences_path) as f:
        fasta_sequences = SeqIO.parse(f, 'fasta')
        sequences_data = np.array([np.array([ltrdict[x]
                                             for x in (str(fasta.seq)).lower()]) for fasta in fasta_sequences])

    with open(labels_path, "r") as f:
        labels = np.array([line.strip() for line in f.readlines()])

    X, y = subsample_data(sequences_data, labels, perc=perc)

    return X, y


def subsample_data(X, y, perc=1.0):
    if not 0.0 <= perc <= 1.0:
        raise ValueError("Illegal Percentage")

    if perc == 1.0:
        return X, y
    else:
        n = len(X)
        indices = np.random.choice(np.arange(0, n), int(n*perc))
        return X[indices], y[indices]


def get_data(experiment, files_path, cell_line, perc):
    d = {'bayesianMLP': import_epigenetic_dataset,
         'bayesianCNN': import_sequence_dataset,
         'fixedCNN': import_sequence_dataset}

    return d[experiment](files_path, cell_line, perc)


def split(X, y, task, random_state=42, test_perc=0.3, proportions=None, mode='u'):
    if mode not in ['u', 'fb', 'b']:
        raise ValueError("Illegal mode value")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_perc, random_state=random_state)

    if mode != 'u':
        X_train, y_train = downsample_data(X_train, y_train, max_size_given=3000)

    if mode == 'fb':
        X_test, y_test = resampling_with_proportion(X_test, y_test, proportions)

    X_train, y_train = filter_by_tasks(X_train, y_train, task)

    X_test, y_test = filter_by_tasks(X_test, y_test, task)

    return X_train, X_test, y_train, y_test


def filter_by_tasks(X, y, task):
    if len(task) != 2:
        raise ValueError("Illegal task dimension")

    new_y = [(i, t["name"]) for t in task for i, label in enumerate(y) if label in t["labels"]]
    res = [[i for i, j in new_y],
           [j for i, j in new_y]]
    indices, y = res[0], res[1]
    X = X[indices]

    return X, y


# TODO: probably useless
def split_2(X, y, random_state=42, proportions=None, mode='u'):
    if mode not in ['u', 'fb', 'b']:
        raise ValueError("Illegal mode value")

    splitters_dict = {'u': unbalanced_splitting,
                      'fb': full_balanced_splitting,
                      'b': balanced_splitting}
    splitter = splitters_dict[mode]

    if proportions:
        return splitter(X, y, random_state=random_state, proportions=proportions)
    else:
        return splitter(X, y, random_state=random_state)


# TODO: add constant for 7
# TODO: testing!!!
def full_balanced_splitting(X, y, test_perc=0.3, random_state=42, proportions=np.array([1, 1, 1, 2, 2, 1, 10])):
    if len(proportions) >= 7:
        raise ValueError("proportion length must be maximum 7 (number of classes)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_perc, random_state=random_state)
    X_train, y_train = downsample_data(X_train, y_train, max_size_given=3000)
    X_test, y_test = resampling_with_proportion(X_test, y_test, proportions)

    return X_train, X_test, y_train, y_test


def balanced_splitting(X, y, test_perc=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_perc, random_state=random_state)
    X_train, y_train = downsample_data(X_train, y_train, max_size_given=3000)

    return X_train, X_test, y_train, y_test


def unbalanced_splitting(X, y, test_perc=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_perc, random_state=random_state)


# TODO: fixing replace in sampling
def get_indices(indices, sample_sizes, n_classes, replace=False):
    indices_range = np.arange(len(indices))
    indices_all = np.concatenate([np.random.choice(indices_range[indices == i],
                                                   size=sample_sizes[i], replace=replace) for i in range(n_classes)])

    return indices_all


def downsample_data(data, classes, max_size_given=None):
    u, indices = np.unique(classes, return_inverse=True)
    num_u = len(u)
    sample_sizes = np.bincount(indices)

    size_max = np.amax(sample_sizes)

    if size_max < max_size_given:
        max_size_given = size_max
    sample_sizes[sample_sizes > max_size_given] = max_size_given

    indices_all = get_indices(indices, sample_sizes, num_u)
    data = data[indices_all, :]
    classes = classes[indices_all]

    return data, classes


def resampling_with_proportion(data, classes, proportions):
    u, indices = np.unique(classes, return_inverse=True)
    num_u = len(u)
    index_max_prop = np.argmax(proportions)
    sizes = np.bincount(indices)
    sample_sizes = [int(round(sizes[index_max_prop] * (prop / proportions[index_max_prop]))) for prop in proportions]

    indices_all = get_indices(indices, sample_sizes, num_u, replace=True)
    data = data[indices_all, :]
    classes = classes[indices_all]

    return data, classes


def encoding_labels(y):
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    return y

