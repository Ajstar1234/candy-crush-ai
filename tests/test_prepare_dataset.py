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