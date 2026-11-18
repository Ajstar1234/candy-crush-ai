import numpy as np
import prepare_dataset

from prepare_dataset import prepare_datasets


def test_prepare_datasets():
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_datasets()

    datasets = [
    (X_train, y_train),
    (X_val, y_val),
    (X_test, y_test)
]

    for X, y in datasets:
        assert X.shape[1] == 180
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] > 0

def test_dataset_seed_ranges():
    train_seeds = set(range(0, 500))
    val_seeds = set(range(500, 520))
    test_seeds = set(range(520, 540))

    assert train_seeds.isdisjoint(val_seeds)
    assert train_seeds.isdisjoint(test_seeds)
    assert val_seeds.isdisjoint(test_seeds)

def test_prepare_datasets_grouping(monkeypatch):

    def fake_generate_training_data(seed=None, return_group_sizes=False, **kwargs):
        X = np.array([[seed, 1], [seed, 2]])
        y = np.array([3, 5])

        if return_group_sizes:
            return X, y, [2]

        return X, y

    monkeypatch.setattr(
        prepare_dataset,
        "generate_training_data",
        fake_generate_training_data
    )

    original = prepare_dataset.prepare_datasets(
        return_group_sizes=False
    )

    grouped = prepare_dataset.prepare_datasets(
        return_group_sizes=True
    )

    assert len(original) == 6
    assert len(grouped) == 9

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        train_groups, val_groups, test_groups
    ) = grouped

    assert sum(train_groups) == len(X_train)
    assert sum(val_groups) == len(X_val)
    assert sum(test_groups) == len(X_test)

    assert np.all(X_train[:, 0] == 0)
    assert np.all(X_val[:, 0] == 500)
    assert np.all(X_test[:, 0] == 520)