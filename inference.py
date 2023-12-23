import math

from preprocessing import read_test, represent_input_with_features
from tqdm import tqdm
from collections import OrderedDict
import numpy as np

f = represent_input_with_features


def get_tri_probability(v, t, u, sentence, k, all_tags, feature2id, pre_trained_weights):
    """ Returns the probability that the tag of the k word is v based on t as the k-2 tag,
        u as the k-1 tag and the current sentence.
        Args:
            v (string): The tag of the Kth positioned word.
            t (string): The tag of the K-2th positioned word.
            u (string): The tag of the K-1th positioned word.
            sentence (list of strings): The current sentence.
            k (int): The position of the requested word.
        Returns:
            numerator/denominator (float): Represent the probability of the requested event.
    """
    linear_term = 0
    # Add ['*'] , ['*'] to the beginning of the sentence
    split_words = ['*'] + ['*'] + sentence

    pp_tag = t
    p_tag = u
    c_tag = v

    # Get all the information for the history
    pp_word, p_word, c_word, n_word = split_words[k - 2:k + 2]
    history = (c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word)

    # Other Histories possible with all tag available
    all_histories = []
    for tag in all_tags:
        all_histories.append((c_word, tag, p_word, p_tag, pp_word, pp_tag, n_word))

    # Get feature available for the word
    word_features_list = f(history, feature2id.feature_to_idx)

    # Run through all the feature for the word and calculate their probability to appear in the sentence
    for feature in word_features_list:
        linear_term += pre_trained_weights[feature]

    # Getting the value of probability of the current tag to exp ( for the sigmoid approx.)
    numerator = math.exp(linear_term)

    # Calculate denominator of the probability ( normalization term )
    denominator = 0

    # Getting the sum of all other history
    for curr_history in all_histories:
        linear_term = 0
        word_features_list = f(curr_history, feature2id.feature_to_idx)
        for feature in word_features_list:
            linear_term += pre_trained_weights[feature]
        denominator = denominator + math.exp(linear_term)
    return float(numerator) / denominator


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """

    # print("Inside viterbi")
    # Getting all the tags available for a word
    all_tags = list(feature2id.feature_to_idx['f105'])

    # Adding the ['*'] tag
    all_tags = ['*'] + all_tags

    # Setting the size of the top beam size ( the best node to continue in the search )
    beam_size = 3

    num_of_words = len(sentence)
    num_of_tags = len(all_tags)
    tags_pairs = [(x, y) for x in all_tags for y in all_tags]
    tags_pair_pos = {(pair[0], pair[1]): i for i, pair in enumerate(tags_pairs)}

    # Init the predicted tag "*" for all the word in the sentence
    predicted_tags = ['*' for word in range(num_of_words)]

    # Init table of π and BP

    pi_table = np.full((num_of_words, num_of_tags ** 2), -np.inf)
    bp_table = np.zeros((num_of_words, num_of_tags ** 2))

    # The first word is tag with "*" for u,v ( ptag,pptag )
    pi_table[1, tags_pair_pos[('*', '*')]] = 1

    # Running through all word
    for i in range(2, num_of_words):

        # Running through all 2 previous tag of the word
        for u, v in tags_pairs:

            # Getting Array of probs of all tags possible
            probs = np.full(len(all_tags), -np.inf)

            # Running through t = tag to find the best one ( with the biggest prob )
            for j, t in enumerate(all_tags):

                if pi_table[i - 1, tags_pair_pos[(t, u)]] == -np.inf or v == '*':
                    continue
                # Get the probability of the trigram given the current history
                trigram_prob = get_tri_probability(v, t, u, sentence, i, all_tags, feature2id, pre_trained_weights)
                probs[j] = pi_table[i - 1, tags_pair_pos[(t, u)]] * trigram_prob

            # Keeping the max prob of the array of prob
            max_var = np.argmax(probs)
            pi_table[i, tags_pair_pos[(u, v)]] = probs[max_var]
            bp_table[i, tags_pair_pos[(u, v)]] = max_var

            # Keeping only the best B candidates
            partitioned = np.partition(pi_table[i, :], len(pi_table[i, :]) - beam_size)
            pi_table[i, :] = np.where(pi_table[i, :] >= partitioned[len(pi_table[i, :]) - beam_size],
                                      pi_table[i, :], -np.inf)

    # Backward step : given the best result at the end we are predicted the other word
    predicted_tags[num_of_words - 2], predicted_tags[num_of_words - 1] = tags_pairs[
        np.argmax(pi_table[num_of_words - 1, :])]
    for i in range(num_of_words - 3, 0, -1):
        predicted_tags[i] = all_tags[int(bp_table[i + 2, tags_pair_pos[(predicted_tags[i + 1],
                                                                        predicted_tags[i + 2])]])]

    predicted_tags = predicted_tags[1:]

    return predicted_tags


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)
    output_file = open(predictions_path, "a+")

    # If the word are already tag , calculate our confusion matrix
    if tagged:
        # Useful variables for our confusion matrix
        all_tags = list(feature2id.feature_to_idx['f105'])
        lengh_all_tags = len(all_tags)
        dict_tags = {i: all_tags[i] for i in range(lengh_all_tags)}
        dict_tags_inv = {v: k for k, v in dict_tags.items()}
        confusion_matrix = np.zeros((lengh_all_tags, lengh_all_tags))

        for k, sen in tqdm(enumerate(test), total=len(test)):
            sentence = sen[0]
            sentence = sentence[2:]
            correct_labels = sen[1]
            correct_labels = correct_labels[2:-2]

            # Run Vitterbi
            pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]

            # Create a confusion matrix
            for i in range(len(pred)):
                predicted = dict_tags_inv[pred[i]]
                try:
                    true = dict_tags_inv[correct_labels[i]]
                    confusion_matrix[predicted, true] += 1
                except KeyError:
                    pass

        # Removing Trace from confusion matrix to select worst ten tags
        for_worst = confusion_matrix - np.diag(np.diag(confusion_matrix))
        # Selecting ten worst tags
        ten_worst = np.argsort(np.sum(for_worst, axis=0))[-10:]
        ten_worst_conf_mat = confusion_matrix[np.ix_(ten_worst, ten_worst)]
        # Computing Accuracy
        accuracy = 100 * np.trace(confusion_matrix) / np.sum(confusion_matrix)
        # print("Model " + str(num_model) + " Accuracy.: " + str(accuracy) + " %")
        print("Accuracy.: " + str(accuracy) + " %")
        print("Ten Worst Elements: " + str([dict_tags[i] for i in ten_worst]))
        print("Confusion Matrix:")
        print(ten_worst_conf_mat)
    else:
        for k, sen in tqdm(enumerate(test), total=len(test)):
            sentence = sen[0]
            sentence = sentence[2:]
            pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
            # sentence = sentence[2:]
            for i in range(len(pred)):
                if i > 0:
                    output_file.write(" ")
                output_file.write(f"{sentence[i]}_{pred[i]}")
            output_file.write(" ._.")
            output_file.write("\n")
        output_file.close()
