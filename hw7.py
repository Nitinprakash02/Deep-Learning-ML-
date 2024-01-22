import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
import urllib.request as download
import plotnine as p9
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class InitialNode:
    def __init__(self, value):
        self.value = value
        self.grad = None

class Operation:
    def __init__(self, *node_list):
        self.input_nodes = node_list
        self.value = None

    def backward(self):
        grad_values = self.gradient()
        for node, grad_value in zip(self.input_nodes, grad_values):
            node.grad = grad_value
            if isinstance(node, Operation):
                node.backward()

class mm(Operation):
    input_names = ('A', 'B')

    def forward(self):
        self.value = np.dot(self.input_nodes[0].value, self.input_nodes[1].value)

    def gradient(self):
        return (np.dot(self.input_nodes[1].value.T, self.input_nodes[0].grad),
                np.dot(self.input_nodes[0].value.T, self.input_nodes[1].grad))


class relu(Operation):
    input_names = ('X',)

    def forward(self):
        self.value = np.maximum(0, self.input_nodes[0].value)

    def gradient(self):
        return (np.where(self.input_nodes[0].value > 0, self.input_nodes[0].grad, 0),)

class logistic_loss(Operation):
    input_names = ('Y_pred', 'Y_true')

    def forward(self):
        scores = self.input_nodes[0].value
        labels = self.input_nodes[1].value
        self.value = -np.mean(labels * np.log(sigmoid(scores)) + (1 - labels) * np.log(1 - sigmoid(scores)))

    def gradient(self):
        return (self.input_nodes[0].value - self.input_nodes[1].value,)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class AutoMLP:
    def __init__(self, max_epochs, batch_size, step_size, units_per_layer, intercept=False):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.units_per_layer = units_per_layer
        self.intercept = intercept

        self.weight_node_list = []
        for i in range(len(units_per_layer) - 1):
            input_size = units_per_layer[i] + int(intercept) 
            output_size = units_per_layer[i + 1]
            weight_matrix = np.random.randn(input_size, output_size) * 0.01
            weight_node = InitialNode(weight_matrix)
            self.weight_node_list.append(weight_node)

    def get_pred_node(self, X):
        input_node = InitialNode(X)

        pred_node = input_node
        for i, weight_node in enumerate(self.weight_node_list):
            pred_node = mm(pred_node, weight_node)
            if i < len(self.weight_node_list) - 1: 
                pred_node = relu(pred_node)

        return pred_node

    def take_step(self, X, y):
        label_node = InitialNode(y)

        pred_node = self.get_pred_node(X)
        loss_node = logistic_loss(pred_node, label_node)

        loss_node.backward()

        for weight_node in self.weight_node_list:
            weight_node.value -= self.step_size * weight_node.grad

    def fit(self, X, y):
        dl = DataLoader(X, y, batch_size=self.batch_size)
        loss_df_list = []
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        for epoch in range(self.max_epochs):
            for batch_features, batch_labels in dl:
                self.take_step(batch_features, batch_labels)
            # TODO: Compute subtrain/validation loss using current weights and append to loss_df_list
            train_loss = self.compute_loss(X_train, y_train)
            val_loss = self.compute_loss(y_train, y_val)
            loss_df_list.append({"epoch": epoch, "train loss": train_loss, "val loss": val_loss})  
        self.loss_df = pd.concat(loss_df_list)

    def compute_loss(self, X, y):
        pred_node = self.get_pred_node(X)
        label_node = InitialNode(y)
        loss_node = logistic_loss(pred_node, label_node)
        loss_node.backward()
        return loss_node.value

    def decision_function(self, X):
        pred_node = self.get_pred_node(X)
        return pred_node.value

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores > 0.5, 1, 0)

class DataLoader:
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_samples = len(X)

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= self.num_samples:
            raise StopIteration
        batch_features = self.X[self.current_index:self.current_index + self.batch_size]
        batch_labels = self.y[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        return batch_features, batch_labels

class AutoGradLearnerCV:
    def __init__(self, max_epochs, batch_size, step_size, units_per_layer):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.units_per_layer = units_per_layer

        self.subtrain_model = AutoMLP(max_epochs=max_epochs,
                                      batch_size=batch_size,
                                      step_size=step_size,
                                      units_per_layer=units_per_layer)

    def fit(self, X, y):
        # Cross-validation for selecting the best number of epochs
        # TODO: Implement cross-validation and find the best number of epochs
        best_epochs = None
        best_accuracy = 0.0
        kf=KFold(n_splits=3)
        for train_indices, test_indices in kf.split(X):
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            subtrain_model = AutoMLP(max_epochs=self.max_epochs,
                                       batch_size=self.batch_size,
                                       step_size=self.step_size,
                                       units_per_layer=self.units_per_layer)
            subtrain_model.fit(X_train, y_train)

            predictions = subtrain_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # TODO: store best epochs

        # TODO: fit model with best epochs

    def predict(self, X):
        return self.train_model.predict(X)

data_dict = {}

spam_data_path = "C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW5/spam_data.csv"
zip_data_path = "C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW5/zip_data.gz"

spam_data = pd.read_csv(spam_data_path, sep=' ', header=None)
X_spam = spam_data.iloc[:, :-1].values
y_spam = spam_data.iloc[:, -1].values
scaler_spam = StandardScaler()
X_spam = scaler_spam.fit_transform(X_spam)

data_dict["spam"] = (X_spam, y_spam)

label_col_num = 0
zip_data = pd.read_csv(zip_data_path, sep=" ", header=None).dropna(axis=1)
label_col_num = int(label_col_num)
data_label_vec = zip_data.iloc[:, label_col_num]
is_01 = data_label_vec.isin([0, 1])
data01_df = zip_data[is_01]
is_label_col = zip_data.columns == label_col_num
zip_features = (data01_df.iloc[:, ~is_label_col].values)
zip_labels = data01_df.iloc[:, is_label_col].values

data_dict["zip"] = (zip_features, zip_labels)

for data_name,(data_scaled,data_labels) in data_dict.items():
    data_dict[data_name]={"X":data_scaled,"y":data_labels}

test_acc_df_list = []

for data_name, data in data_dict.items():
    model_units = {
        "linear": (data["X"].shape[1], 1),
        "deep": (data["X"].shape[1], 100, 10, 1)
    }
    for test_fold, indices in enumerate(KFold(n_splits=3).split(data["X"])):
        X_train, X_test = data["X"][indices[0]], data["X"][indices[1]]
        y_train, y_test = data["y"][indices[0]], data["y"][indices[1]]
        for model_name, units_per_layer in model_units.items():
            # TODO: Fit(train data), then predict(test data), then store accuracy
            learner = AutoGradLearnerCV(100, 32, 0.1, units_per_layer)
            learner.fit(X_train, y_train)
            predictions = learner.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            test_row = {"data_name": data_name, "test_fold": test_fold, "model_name": model_name, "accuracy": accuracy}
            test_acc_df_list.append(test_row)

test_acc_df = pd.DataFrame(test_acc_df_list)

print(test_acc_df)

# Plotting with white background
plot = (p9.ggplot(test_acc_df, p9.aes(x='test_fold', y='accuracy', color='model_name'))
 + p9.facet_wrap('~data_name')
 + p9.geom_line()
 + p9.theme_minimal()  # Original theme
 + p9.theme(panel_background=p9.element_rect(fill='white')) 
 + p9.geom_point()  # Add points to the plot
 + p9.geom_text(p9.aes(label='accuracy'), nudge_y=0.02, color='black', size=8)) 

output_file_path = "C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW7/acc.png"
plot.save(output_file_path)
