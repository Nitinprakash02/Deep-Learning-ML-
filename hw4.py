import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import os
import matplotlib as plt
import urllib.request

plt.use("agg")

class MyKNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
          
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
	        
    def predict(self, X):
        """compute vector of predicted labels."""
        predictions = []
        for x in X:
            distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_labels = [self.y_train[i] for i in nearest_indices]
            most_common = Counter(nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)

class MyCV:
    def __init__(self, estimator, param_grid, cv):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.best_params_ = {}

    def fit_one(self, param_dict, X, y):
        self.estimator.n_neighbors = param_dict["n_neighbors"]
        self.estimator.fit(X, y)
          
    def fit(self, X, y):
        validation_df_list = []
        kf = KFold(n_splits = self.cv, shuffle = True, random_state = 3)
        
        for validation_fold in range(self.cv):
            train_idx, val_idx = next(iter(kf.split(X)))
            subtrain_X, subtrain_y = X[train_idx], y[train_idx]
            val_X, val_y = X[val_idx], y[val_idx]
            
            for param_dict in self.param_grid:
                #self.fit_one(param_dict, **split_data["subtrain"])
                self.fit_one(param_dict, subtrain_X, subtrain_y)
                val_accuracy = self.estimator.score(val_X, val_y) * 100
                validation_row = {
                    'validation_fold': validation_fold,
                    'accuracy_percent': val_accuracy,
                    **param_dict
                }
                validation_df_list.append(validation_row)
        validation_df = pd.concat(validation_df_list)
        best_param_dict = validation_df.groupby(list(self.param_grid[0].keys()))['accuracy_percent'].mean().idxmax()
        self.best_params_ = dict(best_param_dict)
        self.fit_one(best_param_dict, X, y)
          
    def predict(self, X):
        self.estimator.predict(X)

#data files path
spam_data_path = "C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW4/spam_data.csv"
zip_data_path = "c:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW4/zip_data.gz"

#url to download data
spam_data_url = "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data"
zip_data_url = "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.train.gz"

#downloading zip.train data if not exists
if not os.path.exists(zip_data_path):
    urllib.request.urlretrieve(zip_data_url, zip_data_path)
    print("ZIP data file downloaded successfully")
else:
    print("ZIP data file already exists")

#downloading spam data if not exists
if not os.path.exists(spam_data_path):
    urllib.request.urlretrieve(spam_data_url, spam_data_path)
    print("Spam data file downloaded successfully")
else:
    print("Spam data file already exists")

#read data
zip_df = pd.read_csv(zip_data_path, sep=" ", header=None, compression='gzip', dtype=float).dropna(axis=1)
spam_df = pd.read_csv(spam_data_path, sep=" ", header=None, dtype=float)

spam_label_col = -1

#remove rows with non 0/1 labels in spam data
spam_df = spam_df[(spam_df[57] == 0) | (spam_df[57] == 1)]

spam_features = spam_df.iloc[:, :spam_label_col]
spam_labels = spam_df.iloc[:, spam_label_col]

zip_label_col_num = 0
zip_label_vec = zip_df.iloc[:, zip_label_col_num]
is_01 = zip_label_vec.isin([0, 1])
zip01_df = zip_df.loc[is_01, :]
is_label_col = zip_df.columns == zip_label_col_num
zip_features = zip01_df.iloc[:, ~is_label_col]
zip_labels = zip01_df.iloc[:, is_label_col]

data_dict = {
    "spam": (spam_features, spam_labels),
    "zip": (zip_features, zip_labels)
}

#number of folds
K=5

test_acc_df_list = []
kf = KFold(n_splits=K, shuffle=True, random_state=3)
for data_name, (data_features, data_labels) in data_dict.items():
    for test_fold, indices in enumerate(kf.split(data_features)):
        train_idx, test_idx = indices
        train_features, train_labels = data_features.iloc[train_idx], data_labels.iloc[train_idx]
        #test_nrow, test_ncol = test_features.shape
        test_features, test_labels = data_features.iloc[test_idx], data_labels.iloc[test_idx]
        #test_features = test_features.to_numpy().reshape(1, test_ncol)
        
        #instance of MyCV
        my_cv = MyCV(estimator=MyKNN(), param_grid=[{'n_neighbors': n_neighbors} for n_neighbors in range(20)], cv=K)

        #fit the training data
        my_cv.fit(train_features, train_labels)

        #predictions on test data
        test_predictions = my_cv.predict(test_features)

        test_accuracy = accuracy_score(test_labels, test_predictions) * 100

        test_row = {
            'data_set': data_name,
            'test_fold': test_fold,
            'algorithm': 'MyCV+MyKNN',
            'test_accuracy_percent': test_accuracy
        }
        test_acc_df_list.append(test_row)
        
test_acc_df = pd.concat(test_acc_df_list)

print(test_acc_df)
    
