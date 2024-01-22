import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class TorchModel(nn.Module):
    def __init__(self, units_per_layer):
        super(TorchModel, self).__init__()
        layers = []
        for i in range(len(units_per_layer) - 1):
            layers.append(nn.Linear(units_per_layer[i], units_per_layer[i+1]))
            if i < len(units_per_layer) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class TorchLearner:
    def __init__(self, max_epochs, batch_size, step_size, units_per_layer):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.units_per_layer = units_per_layer
        self.model = TorchModel(units_per_layer)
        self.optimizer = optim.SGD(self.model.parameters(), lr=step_size)
        self.loss_fun = nn.BCEWithLogitsLoss()

    def take_step(self, X, y):
        self.optimizer.zero_grad()
        predictions = self.model(X)
        loss = self.loss_fun(predictions, y)
        loss.backward()
        self.optimizer.step()

    def fit(self, X, y, validation_data=None):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        subtrain_losses = []
        validation_losses = []
        for epoch in range(self.max_epochs):
            for batch_features, batch_labels in dataloader:
                self.take_step(batch_features, batch_labels)
            subtrain_predictions = self.model(X)
            subtrain_loss = self.loss_fun(subtrain_predictions, y)
            subtrain_losses.append(subtrain_loss.item())
            if validation_data is not None:
                val_features, val_labels = validation_data
                val_features = torch.tensor(val_features, dtype=torch.float32)
                val_labels = torch.tensor(val_labels, dtype=torch.float32).view(-1, 1)
                val_predictions = self.model(val_features)
                val_loss = self.loss_fun(val_predictions, val_labels)
                validation_losses.append(val_loss.item())
        plt.plot(subtrain_losses, label='Subtrain Loss')
        plt.plot(validation_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def decision_function(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        return self.model(X).detach().numpy()

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores > 0, 1, 0)

class TorchLearnerCV:
    def __init__(self, max_epochs, batch_size, step_size, units_per_layer, validation_data=None):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.units_per_layer = units_per_layer
        self.validation_data = validation_data

    def fit(self, X, y):
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        best_epochs = None
        best_validation_loss = float('inf')
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            model = TorchLearner(max_epochs=self.max_epochs, batch_size=self.batch_size, step_size=self.step_size, units_per_layer=self.units_per_layer)
            model.fit(X_train, y_train, validation_data=(X_val, y_val))
            val_predictions = model.decision_function(X_val)
            val_loss = log_loss(y_val, val_predictions)
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                best_epochs = model.max_epochs
        print(f'Best number of epochs: {best_epochs}')
        final_model = TorchLearner(max_epochs=best_epochs, batch_size=self.batch_size, step_size=self.step_size, units_per_layer=self.units_per_layer)
        final_model.fit(X, y, validation_data=self.validation_data)

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

for data_name, (X, y) in data_dict.items():
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    linear_model = TorchLearnerCV(max_epochs=100, batch_size=32, step_size=0.01, units_per_layer=[X.shape[1], 1], validation_data=(X_val, y_val))
    linear_model.fit(X_train, y_train)

    deep_model = TorchLearnerCV(max_epochs=100, batch_size=32, step_size=0.01, units_per_layer=[X.shape[1], 32, 16, 1], validation_data=(X_val, y_val))
    deep_model.fit(X_train, y_train)
