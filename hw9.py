import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

#matplotlib.use('agg')

class TorchModel(nn.Module):
    def __init__(self, units_per_layer):
        super(TorchModel, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(units_per_layer) - 1):
            self.layers.append(nn.Linear(units_per_layer[i], units_per_layer[i+1]))
            self.layers.append(nn.ReLU())
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        return x

class TorchLearner:
    def __init__(self, max_epochs, batch_size, step_size, units_per_layer):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.units_per_layer = units_per_layer
        self.model = TorchModel(units_per_layer)
        self.optimizer = optim.SGD(self.model.parameters(), lr=step_size)
        self.loss_fun = nn.MSELoss()

    def take_step(self, X, y):
        predictions = self.model(X)

        loss = self.loss_fun(predictions, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def fit(self, X, y, validation_data=None):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        if validation_data:
            X_val, y_val = validation_data
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            for i in range(0, len(X), self.batch_size):
                X_batch = X_tensor[i:i + self.batch_size]
                y_batch = y_tensor[i:i + self.batch_size]

                batch_loss = self.take_step(X_batch, y_batch)
                epoch_loss += batch_loss

            avg_epoch_loss = epoch_loss / (len(X) / self.batch_size)
            print(f"Epoch {epoch + 1}/{self.max_epochs}, Loss: {avg_epoch_loss}")

            if validation_data:
                with torch.no_grad():
                    val_predictions = self.model(X_val_tensor)
                    val_loss = self.loss_fun(val_predictions, y_val_tensor).item()
                    print(f"Validation Loss: {val_loss}")

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.numpy()

class TorchLearnerCV:
    def __init__(self, max_epochs_range, batch_size, step_size, units_per_layer, validation_ratio=0.2):
        self.max_epochs_range = max_epochs_range
        self.batch_size = batch_size
        self.step_size = step_size
        self.units_per_layer = units_per_layer
        self.validation_ratio = validation_ratio
        self.train_loss = []
        self.val_loss = []

    def fit(self, X, y):
        X_subtrain, X_val, y_subtrain, y_val = train_test_split(X, y, test_size=self.validation_ratio, random_state=42)

        best_epochs = None
        best_val_loss = float('inf')

        for max_epochs in self.max_epochs_range:
            learner = TorchLearner(max_epochs, self.batch_size, self.step_size, self.units_per_layer)

            learner.fit(X_subtrain, y_subtrain, validation_data=(X_val, y_val))
            with torch.no_grad():
                train_predictions = learner.model(torch.tensor(X_subtrain, dtype=torch.float32))
                train_loss = learner.loss_fun(train_predictions, torch.tensor(y_subtrain, dtype=torch.float32))
                self.train_loss.append(train_loss.item())

            with torch.no_grad():
                val_predictions = learner.model(torch.tensor(X_val, dtype=torch.float32))
                val_loss = learner.loss_fun(val_predictions, torch.tensor(y_val, dtype=torch.float32)).item()

                self.val_loss.append(val_loss)

            print(f"Validation Loss for {max_epochs} epochs: {val_loss}")

            if val_loss < best_val_loss:
                best_epochs = max_epochs
                best_val_loss = val_loss

        final_learner = TorchLearner(best_epochs, self.batch_size, self.step_size, self.units_per_layer)
        final_learner.fit(X, y)

        return final_learner

import pandas as pd

forest_data_path = "C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW9/forestfires.csv"
airfoil_data_path = "C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW9/airfoil_self_noise.tsv"

forest_df = pd.read_csv(forest_data_path, sep=",", header=None)
airfoil_df = pd.read_csv(airfoil_data_path, sep="\t", header=None)

forest_df = forest_df.drop(0)
forest_df = forest_df.apply(pd.to_numeric, errors='coerce').fillna(0)

airfoil_df = airfoil_df.drop(0)
airfoil_df = airfoil_df.apply(pd.to_numeric, errors='coerce').fillna(0)

print(forest_df)
print(airfoil_df)

def calculate_test_loss(model, X_test, y_test):
    predictions = model.predict(X_test)
    test_loss = mean_squared_error(y_test, predictions)
    return test_loss

# Function to plot training and validation loss

def plot_loss(train_loss, val_loss, title, xlabel, ylabel):
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    #plt.savefig(f"C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW9/{title}.png")
    plt.show()

test_square_loss_row = []

def run_experiment(data_df, model_name, units_per_layer, max_epochs_range, batch_size, step_size, cv_folds=3):
    X = data_df.iloc[:, :-1].values
    y = data_df.iloc[:, -1].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    i=1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        featureless_test_loss = mean_squared_error(y_test, np.full_like(y_test, np.mean(y_train)))
        test_square_loss_row.append({
            "model" : model_name,
            "Algorithm": "featureless",
            "test_square_loss": featureless_test_loss
        })
        print(f"Featureless Test Loss for {model_name}: {featureless_test_loss}")

        knn_reg = KNeighborsRegressor()
        param_grid = {'n_neighbors': [3, 5, 7]}
        knn_reg_cv = GridSearchCV(knn_reg, param_grid, cv=2)
        knn_reg_cv.fit(X_train, y_train)
        knn_test_loss = calculate_test_loss(knn_reg_cv, X_test, y_test)
        test_square_loss_row.append({
            "model" : model_name,
            "Algorithm": "knn+GridSearchCV",
            "test_square_loss": knn_test_loss
        })
        print(f"KNeighborsRegressor Test Loss for {model_name}: {knn_test_loss}")

        lasso_cv = LassoCV()
        lasso_cv.fit(X_train, y_train)
        lasso_test_loss = calculate_test_loss(lasso_cv, X_test, y_test)
        test_square_loss_row.append({
            "model" : model_name,
            "Algorithm": "lasso_cv",
            "test_square_loss": lasso_test_loss
        })
        print(f"LassoCV Test Loss for {model_name}: {lasso_test_loss}")

        torch_linear_learner = TorchLearnerCV(max_epochs_range, batch_size, step_size, units_per_layer)
        linear_model = torch_linear_learner.fit(X_train, y_train)
        linear_test_loss = calculate_test_loss(linear_model, X_test, y_test)
        test_square_loss_row.append({
            "model" : model_name,
            "Algorithm": "TorchLearnerCV_linear",
            "test_square_loss": linear_test_loss
        })
        print(f"TorchLearnerCV Linear Model Test Loss: {linear_test_loss}")

        units_per_layer_deep = [X_train.shape[1], 20, 10, 1]  
        torch_deep_learner = TorchLearnerCV(max_epochs_range, batch_size, step_size, units_per_layer_deep)
        deep_model = torch_deep_learner.fit(X_train, y_train)
        deep_test_loss = calculate_test_loss(deep_model, X_test, y_test)
        test_square_loss_row.append({
            "model" : model_name,
            "Algorithm": "TorchLearnerCV_deep",
            "test_square_loss": deep_test_loss
        })
        print(f"TorchLearnerCV Deep Model Test Loss: {deep_test_loss}")

        plot_loss(torch_linear_learner.train_loss, torch_linear_learner.val_loss,
                  f"TorchLearnerCV Linear Model Loss fold {i}", "Epochs", "Loss")

        plot_loss(torch_deep_learner.train_loss, torch_deep_learner.val_loss,
                  f"TorchLearnerCV Deep Model Loss fold {i}", "Epochs", "Loss")
        i = i + 1

run_experiment(forest_df, 'Forest Fires', [12, 1], range(1, 11), batch_size=32, step_size=0.05)

#run_experiment(airfoil_df, 'Airfoil Self-Noise', [5, 20, 10, 1], range(1, 11), batch_size=32, step_size=0.05)

test_square_loss = pd.DataFrame(test_square_loss_row)
test_square_loss

import plotnine as p9

p = (
    p9.ggplot(test_square_loss, p9.aes(x='test_square_loss', y='Algorithm', color='model')) +
    p9.geom_point() +
    p9.facet_grid('model~.') +
    p9.theme_bw() +
    p9.labs(title='Test Square Loss for Different Models and Algorithms')
)

p.save("C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW9/test_square_loss.png")
