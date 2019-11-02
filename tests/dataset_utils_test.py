import random
import numpy as np
import pytest
from src.dataset_utils import import_epigenetic_dataset, import_sequence_dataset, subsample_data, split, \
    filter_by_tasks, downsample_data, resampling_with_proportion

cell_lines = ["GM12878", "HelaS3", "HepG2", "K562"]
tasks = [[
    {"name": "A-E", "labels": ["A-E"]},
    {"name": "A-P", "labels": ["A-P"]}
  ], [
    {"name": "A-P", "labels": ["A-P"]},
    {"name": "I-P", "labels": ["I-P"]}
  ], [
    {"name": "A-E", "labels": ["A-E"]},
    {"name": "I-E", "labels": ["I-E"]}
  ], [
    {"name": "I-E", "labels": ["I-E"]},
    {"name": "I-P", "labels": ["I-P"]}
  ], [
    {"name": "A-E+A-P", "labels": ["A-E", "A-P"]},
    {"name": "BG", "labels": ["I-E", "I-P", "UK", "A-X", "I-X"]}
  ]]

labels = ["A-E", "I-E", "A-P", "I-P", "A-X", "I-X", "UK"]


@pytest.mark.parametrize('cell_line', cell_lines)
def test_import_epigenetic_dataset_dim(cell_line):
    X, y = import_epigenetic_dataset("tests_files", cell_line)
    assert len(y) == len(X) == 20
    for x in X:
        assert len(x) == 101


@pytest.mark.parametrize('path,cell_line,perc',
                         [("tests_files", "GM", 0.5),
                          ("tests_files", "GM12878", 3),
                          ("tests_files", "GM12878", -1)])
def test_import_epigenetic_dataset_wrong_values(path, cell_line, perc):
    # checking wrong cell lines exception
    with pytest.raises(ValueError):
        import_epigenetic_dataset(path, cell_line, perc)

    # checking wrong percentage
    with pytest.raises(ValueError):
        import_sequence_dataset(path, cell_line, perc)


@pytest.mark.parametrize('path', ['f', 'wrong_path', 'wrong/path/to/file'])
def test_imports_wrong_path(path):
    with pytest.raises(FileNotFoundError):
        import_epigenetic_dataset(path, "GM12878")

    with pytest.raises(FileNotFoundError):
        import_sequence_dataset(path, "GM12878")


@pytest.mark.parametrize('cell_line', cell_lines)
def test_import_sequence_dataset_dim(cell_line):
    X, y = import_sequence_dataset("tests_files", cell_line)

    expected_X = np.array([[[0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]],
                           [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                           [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]])

    assert np.array_equal(expected_X, X)
    assert len(y) == 20


def generate_labels_random(rows=100):
    return np.random.choice(labels, rows)


def generate_labels(n_labels_dict):
    return np.array([k for k, v in n_labels_dict.items() for _ in range(0, v)])


def generate_random_epigenetic_data(rows=100, col=20, d=None):
    if d:
        rows = sum(d.values())

    X = np.array([[random.uniform(0, 30) for _ in range(col)] for _ in range(rows)])

    if d:
        y = generate_labels(d)
    else:
        y = generate_labels_random(rows=rows)

    assert len(X) == len(y)
    assert X.shape == (rows, col)
    return X, y


def generate_random_sequence_data(rows=100, seq_len=20, d=None):
    if d:
        rows = sum(d.values())

    X = np.array([np.eye(5)[np.random.choice(5, seq_len)] for _ in range(rows)])

    if d:
        y = generate_labels(d)
    else:
        y = generate_labels_random(rows=rows)

    assert len(X) == len(y)
    assert X.shape == (rows, seq_len, 5)
    return X, y


@pytest.mark.parametrize('perc', [-5, -3, -1, -0.0001, 1.0001, 3, 5, 6])
def test_subsample_data_wrong_values(perc):
    X, y = generate_random_epigenetic_data()

    with pytest.raises(ValueError):
        subsample_data(X, y, perc=perc)


@pytest.mark.parametrize('perc', np.linspace(0, 1, 11))
def test_subsample_data_shape(perc):
    rows = 100
    col = 20
    X, y = generate_random_epigenetic_data(rows, col)
    X_sampled, y_sampled = subsample_data(X, y, perc=perc)
    assert X_sampled.shape == (int(rows * perc), col)
    assert y_sampled.shape == (int(rows * perc),)


@pytest.mark.parametrize('task', tasks)
def test_filter_task_sequence(task):
    X, y = generate_random_sequence_data(rows=100, seq_len=20)
    expected_task = [t['name'] for t in task]
    new_X, new_y = filter_by_tasks(X, y, task)

    assert all(e in expected_task for e in new_y)

    X, y = generate_random_epigenetic_data(rows=100, col=20)
    expected_task = [t['name'] for t in task]
    new_X, new_y = filter_by_tasks(X, y, task)

    assert all(e in expected_task for e in new_y)


def count_labels(y):
    u, indices = np.unique(y, return_inverse=True)
    return indices, np.bincount(indices)


@pytest.mark.parametrize("X, y, max_size", [*[(*generate_random_epigenetic_data(rows=200, col=101), x)
                                              for x in range(0, 2500, 500)],
                                            *[(*generate_random_sequence_data(rows=200, seq_len=10), x)
                                              for x in range(0, 2500, 500)]
                                            ])
def test_downsampling_data(X, y, max_size):
    X_down, y_down = downsample_data(X, y, max_size_given=max_size)
    indices, counts = count_labels(y_down)
    if max_size == 0:
        assert len(indices) == 0
    else:
        assert len(set(counts)) <= 1
        assert counts[0] <= max_size


def test_downsampling_data_error():
    X, y = generate_random_epigenetic_data(rows=200, col=101)
    with pytest.raises(ValueError):
        downsample_data(X, y, max_size_given=-3)


@pytest.mark.parametrize("task", tasks)
def test_downsampling_data_filtered(task):
    X, y = generate_random_epigenetic_data(rows=1000, col=101)
    X_filtered, y_filtered = filter_by_tasks(X, y, task)
    X_down, y_down = downsample_data(X_filtered, y_filtered, max_size_given=300)
    indices, counts = count_labels(y_down)
    assert len(counts) == 2
    assert len(set(counts)) <= 1
    assert counts[0] <= 300

    X, y = generate_random_sequence_data(rows=1000, seq_len=101)
    X_filtered, y_filtered = filter_by_tasks(X, y, task)
    X_down, y_down = downsample_data(X_filtered, y_filtered, max_size_given=300)
    indices, counts = count_labels(y_down)
    assert len(counts) == 2
    assert len(set(counts)) <= 1
    assert counts[0] <= 300


@pytest.mark.parametrize("sample_len,prop,expected", [(10, [1, 1, 1, 1, 2, 3, 4], [2, 2, 2, 2, 5, 8, 10]),
                                                      (10, [1, 1, 1, 1, 1, 1, 1], [10, 10, 10, 10, 10, 10, 10]),
                                                      (20, [2, 2, 2, 3, 4, 5, 6], [7, 7, 7, 10, 13, 17, 20]),
                                                      (20, [2, 3, 4, 1, 70, 2, 9], [1, 1, 1, 20, 1, 3])])
def test_resampling_with_proportion(sample_len, prop, expected):
    X, y = generate_random_epigenetic_data(rows=100, col=2, d={l: sample_len for l in labels})

    X_resampled, y_resampled = resampling_with_proportion(X, y, proportions=prop)
    indices, counts = count_labels(y_resampled)

    assert all(a == b for a, b in zip(counts, expected))


@pytest.mark.parametrize("sample_len,prop", [(10, [2, 4]),
                                             (10, [1]),
                                             (15, [1, 4, 3])])
def test_resampling_with_proportion_exceptions(sample_len, prop):
    X, y = generate_random_epigenetic_data(rows=100, col=2, d={l: sample_len for l in labels})

    with pytest.raises(ValueError):
        resampling_with_proportion(X, y, proportions=prop)


@pytest.mark.parametrize("mode, perc", [('b', 0.3),
                                        ('u', 0.3),
                                        ('fb', 0.5)])
def test_split(mode, perc):

    X, y = generate_random_sequence_data(rows=100, seq_len=5, d={l: 40 for l in labels})

    X_train, X_test, y_train, y_test = split(X, y, random_state=42, test_perc=perc,
                                             proportions=[2, 2, 2, 3, 4, 5, 6], mode=mode)
    u, indices = np.unique(y_train, return_inverse=True)

    if mode == 'u':
        assert len(X_test) == int(len(X) * perc)
        assert len(y_test) == int(len(X) * perc)

    if mode == 'b' or mode == 'fb':
        assert len(X_test) == len(y_test)
        assert len(X_train) == len(y_train)

        indices, counts = count_labels(y_train)
        assert len(set(counts)) <= 1

    if mode == 'fb':
        expected = [13, 13, 13, 20, 27, 33, 40]
        indices, counts = count_labels(y_test)
        all(a == b for a, b in zip(counts, expected))
