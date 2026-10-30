from generate_dataset import generate_training_data


def prepare_datasets():
    X_train, y_train = generate_training_data(num_games=500, seed=0)

    X_val, y_val = generate_training_data(num_games=20, seed=100)

    X_test, y_test = generate_training_data(num_games=20, seed=120)

    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_datasets()

    print("Training:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Testing:", X_test.shape, y_test.shape)