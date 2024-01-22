import urllib.request
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import plotnine as p9

class MyLogReg:
    def __init__(self, max_iterations=1000, step_size=0.01):
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.coef_ = None
        self.intercept_ = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
    
        for _ in range(self.max_iterations):
            pred_scores = np.dot(X, self.coef_) + self.intercept_
            pred_probs = self.sigmoid(pred_scores)
            gradient = np.dot(X.T, y - pred_probs) / n_samples
            gradient = gradient.reshape(-1)
            self.coef_ += self.step_size * gradient
            y = y.reshape(-1)
            pred_probs = pred_probs.reshape(-1)
            self.intercept_ += self.step_size * np.sum(y - pred_probs) / n_samples
    
    def decision_function(self, X):
        return np.dot(X, self.coef_) + self.intercept_
    
    def predict(self, X, threshold=0.5):
        return np.where(self.sigmoid(self.decision_function(X)) >= threshold, 1, 0)


    
class MyLogRegCV:
    def __init__(self, max_iterations=1000, step_size=0.01, num_splits=3):
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.num_splits = num_splits
        self.scores_ = None
        self.best_iterations = None
        self.lr = None

    # def fit(self, X, y):
        #kf = KFold(n_splits=self.num_splits, shuffle=True, random_state=42)  
    
        # self.scores_ = []

        # for iteration in range(self.max_iterations):
        #     subtrain_loss_sum = 0
        #     validation_loss_sum = 0

        #     for train_index, val_index in kf.split(X):
        #         X_subtrain, X_validation = X[train_index], X[val_index]
        #         y_subtrain, y_validation = y[train_index], y[val_index]

        #         lr = MyLogReg(max_iterations=self.max_iterations, step_size=self.step_size)  # Use the same max_iterations for each iteration
        #         lr.fit(X_subtrain, y_subtrain)

        #         subtrain_loss = log_loss(y_subtrain, lr.predict(X_subtrain))
        #         validation_loss = log_loss(y_validation, lr.predict(X_validation))

        #         subtrain_loss_sum += subtrain_loss
        #         validation_loss_sum += validation_loss

        #     subtrain_loss_avg = subtrain_loss_sum / self.num_splits
        #     validation_loss_avg = validation_loss_sum / self.num_splits

        #     self.scores_.append((self.max_iterations, 'subtrain', subtrain_loss_avg))  # Use self.max_iterations here
        #     self.scores_.append((self.max_iterations, 'validation', validation_loss_avg))  # Use self.max_iterations here

        # self.scores_ = pd.DataFrame(self.scores_, columns=['iteration', 'set_name', 'loss_value'])
        # best_iteration = self.scores_[self.scores_['set_name'] == 'validation']['loss_value'].idxmin()
        # self.best_iterations = self.scores_.iloc[best_iteration]['iteration']

        # self.lr = MyLogReg(max_iterations=self.best_iterations, step_size=self.step_size)
        # self.lr.fit(X, y)
    
    def fit(self, X, y):
        kf = KFold(n_splits=self.num_splits, shuffle=True, random_state=42)
        
        self.scores_ = []

        for iteration in range(self.max_iterations):
            subtrain_loss_sum = 0
            validation_loss_sum = 0
            
            for train_index, val_index in kf.split(X):
                X_subtrain, X_validation = X[train_index], X[val_index]
                y_subtrain, y_validation = y[train_index], y[val_index]
                
                lr = MyLogReg(max_iterations=iteration, step_size=self.step_size)
                lr.fit(X_subtrain, y_subtrain)
                
                subtrain_loss = log_loss(y_subtrain, lr.predict(X_subtrain))
                validation_loss = log_loss(y_validation, lr.predict(X_validation))
                
                subtrain_loss_sum += subtrain_loss
                validation_loss_sum += validation_loss
            
            subtrain_loss_avg = subtrain_loss_sum / self.num_splits
            validation_loss_avg = validation_loss_sum / self.num_splits
            
            self.scores_.append((iteration, 'subtrain', subtrain_loss_avg))
            self.scores_.append((iteration, 'validation', validation_loss_avg))
        
        self.scores_ = pd.DataFrame(self.scores_, columns=['iteration', 'set_name', 'loss_value'])
        best_iteration = self.scores_[self.scores_['set_name'] == 'validation']['loss_value'].idxmin()
        self.best_iterations = self.scores_.iloc[best_iteration]['iteration']
        
        self.lr = MyLogReg(max_iterations=self.best_iterations, step_size=self.step_size)
        self.lr.fit(X, y)
    
    def decision_function(self, X):
        return self.lr.decision_function(X)
    
    def predict(self, X):
        return self.lr.predict(X)

# Load the spam and zip data
spam_data_path = "C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW5/spam_data.csv"
zip_data_path = "c:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW5/zip_data.gz"

spam_df = pd.read_csv(spam_data_path, sep=" ", header=None, dtype=float)
zip_df = pd.read_csv(zip_data_path, sep=" ", header=None, compression='gzip', dtype=float).dropna(axis=1)

# Preprocess the spam data (standardize)
spam_label_col = -1
spam_features = spam_df.iloc[:, :spam_label_col]
spam_labels = spam_df.iloc[:, spam_label_col]

spam_mean = spam_features.mean()
spam_std = spam_features.std()
spam_scaled = (spam_features - spam_mean) / spam_std

# Create instances of MyLogRegCV and fit on the data
lr_cv_spam = MyLogRegCV(max_iterations=300, step_size=5, num_splits=3)
lr_cv_spam.fit(spam_scaled.to_numpy(), spam_labels.to_numpy())

#print(f"Best number of iterations for spam: {lr_cv.best_iterations}")

# Plot subtrain and validation loss for spam using plotnine
spam_loss_df = pd.concat([lr_cv_spam.scores_[lr_cv_spam.scores_['set_name'] == 'subtrain'],
                          lr_cv_spam.scores_[lr_cv_spam.scores_['set_name'] == 'validation']])
spam_loss_df['set_name'] = pd.Categorical(spam_loss_df['set_name'], categories=['subtrain', 'validation'])

p1 = p9.ggplot(spam_loss_df, p9.aes(x='iteration', y='loss_value', color='set_name')) + \
    p9.geom_line() + p9.labs(title='Subtrain and Validation Loss for Spam Data')+\
    p9.geom_point(p9.aes(x='best_iteration', y='loss_value', color='set_name'))

print(p1)

p1.save("C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW5/spam_log_reg.png")


# For zip data (assuming it's already scaled between -1 and 1)
zip_label_col_num = 0
zip_label_vec = zip_df.iloc[:, zip_label_col_num]
is_01 = zip_label_vec.isin([0, 1])
zip01_df = zip_df.loc[is_01, :]
is_label_col = zip_df.columns == zip_label_col_num
zip_features = zip01_df.iloc[:, ~is_label_col]
zip_labels = zip01_df.iloc[:, is_label_col]

print(zip_features)
print(zip_labels)

# Create instances of MyLogRegCV and fit on the data (no need to standardize)
lr_cv_zip = MyLogRegCV(max_iterations=100, step_size=0.01, num_splits=3)
lr_cv_zip.fit(zip_features.to_numpy(), zip_labels.to_numpy())

# Plot subtrain and validation loss for zip using plotnine
zip_loss_df = pd.concat([lr_cv_zip.scores_[lr_cv_zip.scores_['set_name'] == 'subtrain'],
                         lr_cv_zip.scores_[lr_cv_zip.scores_['set_name'] == 'validation']])
zip_loss_df['set_name'] = pd.Categorical(zip_loss_df['set_name'], categories=['subtrain', 'validation'])

p2 = p9.ggplot(zip_loss_df, p9.aes(x='iteration', y='loss_value', color='set_name')) + \
    p9.geom_line() + p9.labs(title='Subtrain and Validation Loss for Zip Data')
    #theme_minimal() + scale_color_manual(values=("#FF5733", "#3380FF"))

print(p2)






