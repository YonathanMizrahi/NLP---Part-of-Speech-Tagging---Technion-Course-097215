from preprocessing import preprocess_train
import pickle
from optimization import get_optimal_vector
from inference import tag_all_test

# Relevant paths, threshold and lambda for comp1
#train_path = "data/train1.wtag"
#test_path = "data/comp1.words"
#predictions_path = 'comp_m1_209948728_931188684.wtag'
#threshold = 1
#lam = 0.4

# Relevant paths, threshold and lambda for comp2
train_path = "data/train2.wtag"
test_path = "data/comp2.words"
predictions_path = 'comp_m2_209948728_931188684.wtag'
threshold = 1
lam = 0.3

weights_path = 'weights.pkl'

statistics, feature2id = preprocess_train(train_path, threshold)
get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

with open(weights_path, 'rb') as f:
 optimal_params, feature2id = pickle.load(f)
pre_trained_weights = optimal_params[0]
#tag_all_test('test_path', pre_trained_weights, feature2id, 'predictions_path')
tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)


