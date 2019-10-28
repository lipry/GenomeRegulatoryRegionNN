import random
import numpy as np
import pytest
from src.dataset_utils import import_epigenetic_dataset, import_sequence_dataset, subsample_data, split

cell_lines = ["GM12878", "HelaS3", "HepG2", "K562"]


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
    # cheking wrong path
    with pytest.raises(FileNotFoundError):
        import_epigenetic_dataset("f", "GM12878")

    with pytest.raises(FileNotFoundError):
        import_sequence_dataset("f", "GM12878")


@pytest.mark.parametrize('cell_line', cell_lines)
def test_import_sequence_dataset_dim(cell_line):
    X, y = import_sequence_dataset("tests_files", cell_line)

    expected_X = np.array([[[0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]],
                           [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                           [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]])

    assert np.array_equal(expected_X, X)
    assert len(y) == 20


def generate_random_epigenetic_data(rows=100, col=20):
    X = np.array([[random.uniform(0, 30) for _ in range(col)] for _ in range(rows)])

    labels = ["A-E", "I-E", "A-P", "I-P", "A-X", "I-X", "UK"]
    y = np.array([np.random.choice(labels, col) for _ in range(rows)])
    assert len(X) == len(y)
    assert X.shape == (rows, col)
    return X, y
    

@pytest.mark.parametrize('perc', [-5, -3, -1, -0.0001, 1.0001, 3, 5, 6])
def test_subsample_data_wrong_values(perc):
    X, y = generate_random_epigenetic_data()

    with pytest.raises(ValueError):
        subsample_data(X, y, perc=-3)


@pytest.mark.parametrize('perc', np.linspace(0, 1, 11))
def test_subsample_data_shape(perc):
    X, y = generate_random_epigenetic_data()
    X_sampled, y_sampled = subsample_data(X, y, perc=perc)
    assert X_sampled.shape == (int(rows * perc), col)
    assert y_sampled.shape == (int(rows * perc),)


def test_filter_task():
    pass

#def test_split():
#    X, y = generating_random_data()
#    X = split(X, y, task, random_state=42, test_perc=0.3, proportions=None, mode='u')

