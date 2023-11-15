import pickle
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from itertools import combinations

wine = pickle.load(open('wine.pkl', 'rb'))

X = np.array(wine["data"]).astype(float)
Y = np.array(wine["labels"])
X_test = np.array(wine["testdata"]).astype(float)
Y_test = np.array(wine["testlabels"])

def genCombinations(num_features):
    return list(combinations(range(X.shape[1]), num_features))

accuracy_rates = {}

for n in [1, 2]:
    feature_combinations = genCombinations(n)
    for c in feature_combinations:
        X_feature = X[:, c]
        model = GaussianNB()
        scores = cross_val_score(model, X_feature, Y)
        accuracy_rates[np.mean(scores)] = c

highest_accuracy = max(accuracy_rates.keys())
best_features = accuracy_rates[highest_accuracy]


print(f'Optimal features determined by cross-validation: {best_features}')
print(f'Validation error rate: {1-highest_accuracy}\n')

best_model = GaussianNB()
X_best = X[:, best_features]
best_model.fit(X_best, Y)

test_error = 1 - best_model.score(X_test[:,best_features], Y_test)
training_error = 1 - best_model.score(X[:best_features], Y)

print("PART B")
print(f'The training error rate is: {training_error}')
print(f'The test error rate is: {test_error}')
print(f'Class 1 Parameters: mu {best_model.theta_[0]}, variance {best_model.var_[0]}')
print(f'Class 2 Parameters: mu {best_model.theta_[1]}, variance {best_model.var_[1]}')
print(f'Class 3 Parameters: mu {best_model.theta_[2]}, variance {best_model.var_[2]}')