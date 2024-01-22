import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import torchvision
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


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
        self.loss_fun = nn.CrossEntropyLoss()

    def take_step(self, X, y):
        # Compute predictions
        predictions = self.model(X)

        # Compute the mean loss
        loss = self.loss_fun(predictions, y)

        # Zero gradients, backward pass, and optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fit(self, X, y, X_val=None, y_val=None):
        train_losses = []
        val_losses = []

        for epoch in range(self.max_epochs):
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i + self.batch_size]
                batch_y = y[i:i + self.batch_size]

                # Convert to PyTorch tensors
                batch_X = torch.tensor(batch_X, dtype=torch.float32)
                batch_y = torch.tensor(batch_y, dtype=torch.long)

                # Take a step using the current batch
                self.take_step(batch_X, batch_y)

            # Compute and store the training loss for this epoch
            train_predictions = self.model(torch.tensor(X, dtype=torch.float32))
            #y_tensor = torch.tensor(y, dtype=torch.float32)
            train_loss = self.loss_fun(train_predictions, torch.tensor(y, dtype=torch.long))
            train_losses.append(train_loss.item())

            # Compute and store the validation loss if validation data is provided
            if X_val is not None and y_val is not None:
                val_predictions = self.model(torch.tensor(X_val, dtype=torch.float32))
                val_loss = self.loss_fun(val_predictions, torch.tensor(y_val, dtype=torch.long))
                val_losses.append(val_loss.item())

        return train_losses, val_losses

    def decision_function(self, X):
        # Compute and return the matrix of predicted scores
        predictions = self.model(X)
        return predictions

    def predict(self, X):
        # Get the predicted scores
        scores = self.decision_function(X)

        # Return the class with the maximum score for each row
        _, predicted_classes = torch.max(scores, 1)
        return predicted_classes.numpy()

class TorchLearnerCV:
    def __init__(self, max_epochs, batch_size, step_size, units_per_layer, validation_size=0.2):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.units_per_layer = units_per_layer
        self.validation_size = validation_size
        self.best_epochs = None

    def fit(self, X, y):
        # Split the data into subtrain and validation sets
        X_subtrain, X_val, y_subtrain, y_val = train_test_split(X, y, test_size=self.validation_size, random_state=42)

        # Instantiate TorchLearner
        self.learner = TorchLearner(max_epochs=self.max_epochs, batch_size=self.batch_size, step_size=self.step_size,
                               units_per_layer=self.units_per_layer)

        # Run gradient descent on subtrain and validation sets
        train_losses, val_losses = learner.fit(X_subtrain, y_subtrain, X_val, y_val)

        # Find the epoch with the minimum validation loss
        best_epoch = np.argmin(val_losses)

        # Store the best number of epochs
        self.best_epochs = best_epoch + 1  # Adding 1 because epochs are 0-indexed

        # Re-run gradient descent on the entire training set with the best number of epochs
        learner_final = TorchLearner(max_epochs=self.best_epochs, batch_size=self.batch_size, step_size=self.step_size,
                                     units_per_layer=self.units_per_layer)
        final_train_losses, _ = learner_final.fit(X, y)

        return final_train_losses

zip_data_path = "c:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW5/zip_data.gz"

data_dict = {}

data_df = pd.read_csv(zip_data_path, sep=" ", header=None)
label_col_num = 57
data_label_vec = data_df.iloc[:, label_col_num]
is_01 = data_label_vec.isin([0, 1])
data01_df = data_df[is_01]
is_label_col = data_df.columns == label_col_num
data_features = data01_df.iloc[:, ~is_label_col].dropna(axis=1)
data_labels = data01_df.iloc[:, is_label_col]
data_dict["zip"] = (data_features, data_labels)

ds = torchvision.datasets.MNIST(
    root="c:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW5",
    download=True,
    transform=torchvision.transforms.ToTensor(),
    train=False)

dl = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)

for mnist_features, mnist_labels in dl:
    pass

mnist_features = mnist_features.flatten(start_dim=1).numpy()
mnist_labels = mnist_labels.numpy()

data_dict["mnist"] = (mnist_features, mnist_labels)

def plot_losses(train_losses, val_losses, title):
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()

# Function to perform k-fold cross-validation
def kfold_cross_validation(X, y, model, k=3):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        # Train the model using TorchLearnerCV
        learner_cv = TorchLearnerCV(max_epochs=10, batch_size=32, step_size=0.01, units_per_layer=model)
        train_losses = learner_cv.fit(X_train, y_train)

        # Plot training and validation losses
        plot_losses(train_losses, learner_cv.learner.losses, f'Model: {model}, Fold: {len(accuracies) + 1}')

        # Store the best number of epochs
        best_epochs = learner_cv.learner.best_epochs

        # Train the final model on the entire training set
        final_learner = TorchLearner(max_epochs=best_epochs, batch_size=32, step_size=0.01, units_per_layer=model)
        final_learner.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = final_learner.predict(X_val)

        # Calculate accuracy and store it
        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)

    return accuracies

# Function to calculate and plot test accuracy
def plot_test_accuracy(models, data_dict):
    for model_name, (X, y) in data_dict.items():
        accuracies = []

        for model in models:
            learner_cv = TorchLearnerCV(max_epochs=10, batch_size=32, step_size=0.01, units_per_layer=model)
            train_losses = learner_cv.fit(X, y)

            # Train the final model on the entire training set
            final_learner = TorchLearner(max_epochs=learner_cv.learner.best_epochs, batch_size=32, step_size=0.01, units_per_layer=model)
            final_learner.fit(X, y)

            # Make predictions on the test set
            y_pred = final_learner.predict(X_test)

            # Calculate accuracy and store it
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        # Plot the test accuracies
        plt.bar(models, accuracies)
        plt.xlabel('Model')
        plt.ylabel('Test Accuracy')
        plt.title(f'Test Accuracy for {model_name.capitalize()}')
        plt.show()

# Example usage with linear model (no hidden layers)
linear_model = [data_dict["mnist"][0].shape[1], 10]
linear_accuracies = kfold_cross_validation(data_dict["mnist"][0], data_dict["mnist"][1], linear_model)

# Example usage with deep neural network (at least two hidden layers)
deep_model = [input_size, hidden_size1, hidden_size2, 1]
deep_accuracies = kfold_cross_validation(data_dict["zip"][0], data_dict["zip"][1], deep_model)

# Example usage to plot test accuracy
models_to_test = [linear_model, deep_model]
plot_test_accuracy(models_to_test, data_dict)
