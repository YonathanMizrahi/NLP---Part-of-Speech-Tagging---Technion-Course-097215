import pickle
import random
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test


def split_train_test():
    train2_path = "data/train2.wtag"
    train2 = []
    test2 = []
    with open(train2_path) as file:
        print(type(file))
        for line in file:
            if random.uniform(0, 1) >= 0.25:
                train2.append(line)
            else:
                test2.append(line)

    with open("data_for_train2/train2_splitted.wtag", "w") as txt_file:
        for line in train2:
            txt_file.write("".join(line))  # works with any number of elements in a line

    with open("data_for_train2/test2_splitted.wtag", "w") as txt_file:
        for line in test2:
            txt_file.write("".join(line))  # works with any number of elements in a line


def main():

    #split_train_test()

    threshold = 1
    lam = 0.3

    train_path = "data_for_train2/train2_splitted.wtag"
    test_path = "data_for_train2/test2_splitted.wtag"

    weights_path = 'weights.pkl'
    predictions_path = 'predictions.wtag'

    statistics, feature2id = preprocess_train(train_path, threshold)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]
    print(pre_trained_weights)
    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)


if __name__ == '__main__':
    main()