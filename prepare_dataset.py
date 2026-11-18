from generate_dataset import generate_training_data



def prepare_datasets(
        rows=5,
        cols=5,
        num_candy_types=4,
        use_target_mask=False,
        return_group_sizes=False):

    train_data = generate_training_data(
        num_games=500,
        seed=0,
        rows=rows,
        cols=cols,
        num_candy_types=num_candy_types,
        use_target_mask=use_target_mask,
        return_group_sizes=return_group_sizes
    )

    val_data = generate_training_data(
        num_games=20,
        seed=500,
        rows=rows,
        cols=cols,
        num_candy_types=num_candy_types,
        use_target_mask=use_target_mask,
        return_group_sizes=return_group_sizes
    )

    test_data = generate_training_data(
        num_games=20,
        seed=520,
        rows=rows,
        cols=cols,
        num_candy_types=num_candy_types,
        use_target_mask=use_target_mask,
        return_group_sizes=return_group_sizes
    )

    if return_group_sizes:
        X_train, y_train, train_groups = train_data
        X_val, y_val, val_groups = val_data
        X_test, y_test, test_groups = test_data

        return (
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            train_groups, val_groups, test_groups
        )

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

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