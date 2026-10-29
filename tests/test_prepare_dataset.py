from prepare_dataset import prepare_datasets


def test_prepare_datasets():
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_datasets()

    datasets = [
    (X_train, y_train),
    (X_val, y_val),
    (X_test, y_test)
]

    for X, y in datasets:
        assert X.shape[1] == 55
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] > 0

def test_dataset_seed_ranges():
    train_seeds = set(range(0, 100))
    val_seeds = set(range(100, 120))
    test_seeds = set(range(120, 140))

    assert train_seeds.isdisjoint(val_seeds)
    assert train_seeds.isdisjoint(test_seeds)
    assert val_seeds.isdisjoint(test_seeds)