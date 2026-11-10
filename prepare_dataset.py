from generate_dataset import generate_training_data


def prepare_datasets(rows=5, cols=5, num_candy_types=4, use_target_mask=False):
    X_train, y_train = generate_training_data(
    num_games=500,
    seed=0,
    rows=rows,
    cols=cols,
    num_candy_types=num_candy_types,
    use_target_mask=use_target_mask
)

    X_val, y_val = generate_training_data(
    num_games=20,
    seed=500,
    rows=rows,
    cols=cols,
    num_candy_types=num_candy_types,
    use_target_mask=use_target_mask
)

    X_test, y_test = generate_training_data(
    num_games=20,
    seed=520,
    rows=rows,
    cols=cols,
    num_candy_types=num_candy_types,
    use_target_mask=use_target_mask
)

    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_datasets(
        rows=9,
        cols=9,
        num_candy_types=6,
        use_target_mask=True
    )

    print("Training:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Testing:", X_test.shape, y_test.shape)