#%% VS Code Notebook
# Copyright 2022 CMPUT 466 Staff
# Implement Naive Bayes Classifier with Laplace Smoothing
import numpy as np
import pandas as pd

from pathlib import Path

project_root = Path(__file__).parent

# %%
class NaiveBayesClassifier:
    def __init__(self, num_classes: int, num_features: int, smooth: bool = False):
        self.num_classes = num_classes
        self.num_features = num_features
        self.smooth = smooth
    
    def train(self, X: np.array, y: np.array) -> None:
        """
        Updates the self.log_priors and self.log_likelihoods
        based on the features "X" and labels "y" passed in
        """
        self.log_priors = np.zeros(self.num_classes) # Stores P(c)
        self.log_likelihoods = np.zeros((self.num_classes,self.num_features,np.max(X)+1)) # Stores P(e_n|c)
        # !!!!!!!!!!!! Start of your code here !!!!!!!!!!!!
        log_priors_shape = np.shape(self.log_priors)
        for i in range(self.num_classes):
            self.log_priors[i] = np.log(np.count_nonzero(y == i)) - np.log(len(y))
        assert(np.shape(self.log_priors) == log_priors_shape)
        
        log_likelihoods_shape = np.shape(self.log_likelihoods)
        for i in range(len(X)):
            for j in range(len(X[i])):
                self.log_likelihoods[y[i]][j][X[i][j]] += 1
        
        for i in range(self.num_classes):
            self.log_likelihoods[i] = np.log(self.log_likelihoods[i] + 1)
            for j in range(self.num_features):
                self.log_likelihoods[i][j] -= np.log(np.count_nonzero(y == i) + np.count_nonzero(self.log_likelihoods[i][j] > 1))
        assert(np.shape(self.log_likelihoods) == log_likelihoods_shape)
        # !!!!!!!!!!!! End of your code here   !!!!!!!!!!!!

    def predict_classes(self, X: np.array) -> np.array:
        """
        Returns a numpy array with the ids for most probable class
        for each feature row in "X"
        """
        logs_probs = np.zeros((len(X),self.num_classes))
        # !!!!!!!!!!!! Start of your code here !!!!!!!!!!!!
        predictions = np.zeros(len(X))
        for x in range(len(X)):
            for y in range(self.num_classes):
                for e in range(len(X[x])):
                    logs_probs[x][y] += self.log_likelihoods[y][e][X[x][e]] + self.log_priors[y]
            predictions[x] = logs_probs[x].argmax(axis=0)
        return predictions
        # !!!!!!!!!!!! End of your code here   !!!!!!!!!!!!

# %%
col_feature_map = {

}

df_train = pd.read_csv(project_root/'datasets/heart/heart_train.csv')
df_test = pd.read_csv(project_root/'datasets/heart/heart_test.csv')

for col in df_train.columns:
    unique_vals = df_train[col].unique()
    col_feature_map[col] = {val: i for i, val in enumerate(unique_vals)}
    df_train[f'num_{col}'] = df_train[col].map(col_feature_map[col])
    df_test[f'num_{col}'] = df_test[col].map(col_feature_map[col])
# %%
label_column = 'num_HeartDisease'
features_columns = [col for col in df_train.columns if col.startswith('num_') and col != label_column]
# %%
classifier = NaiveBayesClassifier(
    num_classes=len(col_feature_map[label_column.replace('num_','')]), 
    num_features=len(features_columns),
    smooth=True,
)
# %%
classifier.train(
    X=df_train[features_columns].to_numpy(),
    y=df_train[label_column].to_numpy()
)
# %%
df_test['prediction'] = classifier.predict_classes(
    X=df_test[features_columns].to_numpy()
)
# %%
accuracy = np.sum(df_test['prediction'] == df_test[label_column]) / len(df_test)
print(f'Accuracy: {accuracy}')
# %%
