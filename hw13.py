import torch
import torchvision

ds = torchvision.datasets.MNIST(
    root="C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW13",
    download=True,
    transform=torchvision.transforms.ToTensor(),
    train=False)
dl = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)

for mnist_features, mnist_labels in dl:
    pass

Nobs, Nchan, height, width = mnist_features.shape
N_hidden_chan = 20
N_classes=10
conv_seq_obj = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=Nchan, out_channels=N_hidden_chan, kernel_size=4),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=5),
    torch.nn.Flatten())
ignored, n_hidden_after_flatten = conv_seq_obj(mnist_features[:1]).shape
n_linear_hidden = 50
lin_seq_obj = torch.nn.Sequential(
    torch.nn.Linear(n_hidden_after_flatten, n_linear_hidden),
    torch.nn.ReLU(),
    torch.nn.Linear(n_linear_hidden, N_classes)
)
full_seq_obj = torch.nn.Sequential(
    conv_seq_obj,
    lin_seq_obj)

#torch.nn.Conv2d(in_channels=N_hidden_chan, out_channels=10, kernel_size=5))

seq_out_tensor = full_seq_obj(mnist_features)
seq_out_tensor.shape

loss_fun = torch.nn.CrossEntropyLoss()
loss_tensor = loss_fun(seq_out_tensor, mnist_labels)

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split
import pandas as pd

data_df = pd.read_csv("C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW11/zip.test.gz", sep=" ", header=None)

data_size = len(data_df)
data_size

train_size = int(0.8*data_size)
val_size = data_size-train_size

train_data, val_data = random_split(data_df, [train_size, val_size])

batch_size=64

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

data_features = data_df.loc[:, 1:]
data_labels = data_df.loc[:, 1]

class ConvolutionalMLP(nn.Module):
    def __init__(self, input_shape=(1, 16, 16), num_classes=2, dropout=False):
        super(ConvolutionalMLP, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),  # Adjust the input size based on the chosen input_shape
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        if dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.conv_layers(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear_layers(x)
        return x

    def fit(self, train_loader, val_loader, num_epochs=10, lr=0.001):
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                #inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Plotting the loss vs number of epochs
        plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss', color='red')
        plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def predict(self, test_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()

        predictions = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = self(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())

        return predictions

# Example usage:
# Assume you have data_dict, train_loader, and val_loader prepared from your dataset
# You may need to adjust the input_shape based on your actual data dimensions
model = ConvolutionalMLP(input_shape=(1, 16, 16), num_classes=2, dropout=True)
model.fit(train_loader, val_loader, num_epochs=10, lr=0.001)
predictions = model.predict(test_loader)
test_accuracy = accuracy_score(test_labels, predictions)
print(f"Test Accuracy: {test_accuracy}")

import numpy as np
import torch
import math
import matplotlib
import pandas as pd
import torchvision
import plotnine as p9
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

matplotlib.use('agg')

data_info_dict = {
    "zip" : ("c:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW11/zip.test.gz", 0),
    #"spam" : ("C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW5/spam_data.csv", 57)
}
data_dict={}

data_for_lr_knn = []

for data_name, (file_name, label_col_num) in data_info_dict.items():
    data_df = pd.read_csv(file_name, sep = " ", header = None)
    #print(f"{data_name}\n{data_df}")
    data_label_vec = data_df.iloc[:, label_col_num]
    is_label_col = data_df.columns == label_col_num
    data_features = data_df.iloc[:, ~is_label_col]
    data_labels = data_df.iloc[:, is_label_col]
    data_for_lr_knn.append((data_features, data_labels))
    if data_name == "spam":
        X_mean = data_features.mean()
        X_std = data_features.std()
        data_features = (data_features - X_mean) / X_std
    image_size = int(math.sqrt(data_features.shape[1]))
    data_features = data_features.to_numpy().reshape(-1, 1, image_size, image_size)
    print("%s %s"%(data_name, data_features.shape))
    data_dict[data_name] = (
        torch.from_numpy(data_features).float(),
        torch.from_numpy(data_labels.to_numpy()).flatten()
    )

{data_name:features.shape for data_name, (features, labels) in data_dict.items()}

input_tensor, output_tensor = data_dict["zip"]
batch_size, channels, height, width = data_dict["zip"][0].shape

class ConvolutionalMLP(torch.nn.Module):
    def __init__(self, units_per_layer):
        super(ConvolutionalMLP, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(3, 3)),
            torch.nn.Conv2d(32, 64, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = (3, 3)),
            torch.nn.Flatten(start_dim = 1))
        self.lin_seq = torch.nn.Sequential(
            #torch.nn.Linear(1, 128),
            torch.nn.Linear(64 * 16, 128),
            torch.nn.ReLU(),
            #torch.nn.Linear(128, 2)),
            torch.nn.Linear(128, 10))  
        self.fc = torch.nn.Sequential(
            self.conv, 
            self.lin_seq)
        
    def forward(self, features):
        _, flattened_size = self.conv(features[:1]).shape
        self.lin_seq[0] = torch.nn.Linear(flattened_size, 128)
        x = self.fc(features)
        return x
    
n_classes = 10
#n_classes = 2                   

class CSV(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __getitem__(self, item):
        return self.features[item, :], self.labels[item]
    def __len__(self):
        return len(self.labels)

class ConvolutionalLearner:
    def __init__(
            self, units_per_layer,
            batch_size = 20, max_epochs = 100):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.model = ConvolutionalMLP(units_per_layer)
        self.loss_fun = torch.nn.CrossEntropyLoss()
        self.initial_step_size=0.5
        self.end_step_size=0.005
        self.last_step_number=50
    def get_step_size(self, iteration):
        if iteration > self.last_step_number:
            return self.end_step_size
        prop_to_last_step = iteration/self.last_step_number
        return (1-prop_to_last_step)*self.initial_step_size + \
            prop_to_last_step*self.end_step_size
    def fit(self, split_data_dict):
        ds = CSV(
            split_data_dict["subtrain"]["X"],
            split_data_dict["subtrain"]["y"])
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=True)
        train_df_list = []
        for epoch_number in range(self.max_epochs):
            step_size = self.get_step_size(epoch_number)
            print(f"epoch={epoch_number} step_size={step_size}")
            for batch_features,  batch_labels in dl:
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    momentum=0.5,
                    lr=step_size)
                self.optimizer.zero_grad()
                loss_value = self.loss_fun(
                    self.model(batch_features),
                    batch_labels)
                loss_value.backward()
                self.optimizer.step()
            for set_name, set_data in split_data_dict.items():
                pred_vec = self.model(set_data["X"])
                set_loss_value = self.loss_fun(pred_vec, set_data["y"])
                train_df_list.append(pd.DataFrame({
                    "set_name" : [set_name],
                    "loss" : float(set_loss_value),
                    "epoch" : [epoch_number]
                }))
        self.train_df = pd.concat(train_df_list)
    def predict(self, test_features):
        test_pred_scores = self.model(test_features)
        return test_pred_scores.argmax(axis = 1)

class ConvolutionalLearnerCV:
    def __init__(self, n_folds = 3, units_per_layer = [height*width, 1]):
        self.units_per_layer = units_per_layer
        self.n_folds = n_folds
    def fit(self, train_features, train_labels):
        train_nrow, channels, train_height, train_width = train_features.shape
        train_ncol = train_height*train_width
        times_to_repeat = int(math.ceil(train_nrow/self.n_folds))
        fold_id_vec = np.tile(torch.arange(self.n_folds), times_to_repeat)[:train_nrow]
        np.random.shuffle(fold_id_vec)
        cv_data_list = []
        for validation_fold in range(self.n_folds):
            is_split = {
                "subtrain" : fold_id_vec != validation_fold,
                "validation" : fold_id_vec == validation_fold
            }
            split_data_dict = {}
            for set_name, is_set in is_split.items():
                set_y = train_labels[is_set]
                split_data_dict[set_name] = {
                    "X" : train_features[is_set, :],
                    "y" : set_y
                }
            learner = ConvolutionalLearner(self.units_per_layer)
            learner.fit(split_data_dict)
            cv_data_list.append(learner.train_df)
        self.cv_data = pd.concat(cv_data_list)
        self.train_df = self.cv_data.groupby(
            ["set_name", "epoch"]
        ).mean().reset_index()
        valid_df = self.train_df.query("set_name=='validation'")
        #print(valid_df)
        # if not valid_df.empty:
        best_epochs = valid_df["loss"].argmin()
        self.min_df = valid_df.query("epoch==%s"%best_epochs)
        self.final_learner = ConvolutionalLearner(
            self.units_per_layer, max_epochs=best_epochs)
        self.final_learner.fit({"subtrain":{"X":train_features, "y":train_labels}})
    def predict(self, test_features):
        return self.final_learner.predict(test_features)

class MyCV:
    def __init__(self, estimator, param_grid, cv):
        self.cv = cv
        self.param_grid = param_grid
        self.estimator = estimator
    def fit_one(self, param_dict, X, y):
        for param_name, param_value in param_dict.items():
            setattr(self.estimator, param_name, param_value)
        self.estimator.fit(X, y)
    def fit(self, X, y):
        validation_df_list = []
        train_nrow, train_ncol = X.shape
        times_to_repeat = int(math.ceil(train_nrow/self.cv))
        fold_id_vec = np.tile(np.arange(self.cv), times_to_repeat)[:train_nrow]
        np.random.shuffle(fold_id_vec)
        for validation_fold in range(self.cv):
            #print(validation_fold)
            is_split = {
                "subtrain" : fold_id_vec != validation_fold,
                "validation" : fold_id_vec == validation_fold
            }
            split_data_dict = {}
            for set_name, is_set in is_split.items():
                #set_y = y[is_set].reshape(-1,1)  
                split_data_dict[set_name] = (
                    X[is_set],
                    y[is_set]
                    #set_y
                )
            for param_number, param_dict in enumerate(self.param_grid):
                self.fit_one(param_dict, *split_data_dict["subtrain"])
                X_valid, y_valid = split_data_dict["validation"]
                pred_valid = self.estimator.predict(X_valid)
                is_correct = pred_valid == y_valid
                validation_row = pd.DataFrame({
                    "validation_fold" : validation_fold,
                    "accuracy_percent" : float(is_correct.float().mean()),
                    "param_number" : [param_number]
                }, index=[0])
                validation_df_list.append(validation_row)
        self.validation_df = pd.concat(validation_df_list)
        self.mean_valid_acc = self.validation_df.groupby(
            "param_number"
        )["accuracy_percent"].mean()
        best_index = self.mean_valid_acc.argmax()
        #mean_valid_acc.index[best_index]
        self.best_param_dict = self.param_grid[best_index]
        self.fit_one(self.best_param_dict, X, y)
    def predict(self, X):
        return self.estimator.predict(X)

accuracy_row = []

for data_name, (data_features, data_labels) in data_dict.items():
    X_train, X_test, y_train, y_test = train_test_split(data_features, data_labels, test_size=0.2, random_state=42)
    #input_tensor, output_tensor = data_dict[data_name]
    input_tensor, output_tensor = X_train, y_train
    #print(f"{input_tensor.shape} {output_tensor.shape}")
    input_tensor_test, output_tensor_test = X_test, y_test
    #data_nrow, data_ncol = input_tensor.shape
    hyper_params = []

    #Logistic regression
    for (data_f, data_l) in data_for_lr_knn:
        pass
    data_f_train, data_f_test, data_l_train, data_l_test = train_test_split(data_f, data_l, test_size=0.2, random_state=42)
    clf = LogisticRegressionCV(cv=3, random_state=0)
    clf.fit(data_f_train, data_l_train)
    lr_predictions = clf.predict(data_f_test)
    lr_accuracy = accuracy_score(lr_predictions, data_l_test)

    accuracy_row.append({
        "dataset" : data_name,
        "accuracy" : lr_accuracy,
        "model" : "LogisticReqgressionCV"
    })

    #kneighbors
    knn = KNeighborsClassifier()
    
    k_range = list(range(1, 10))
    param_grid = dict(n_neighbors=k_range)

    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False, verbose=1)

    grid_search = grid.fit(data_f_train, data_l_train)

    knn = KNeighborsClassifier(n_neighbors = grid_search.best_params_['n_neighbors'])
        
    knn.fit(data_f_train, data_l_train)
    knn_predictions = knn.predict(data_f_test)
    knn_accuracy = accuracy_score(knn_predictions, data_l_test)

    accuracy_row.append({
        "dataset" : data_name,
        "accuracy" : knn_accuracy,
        "model" : "GridSearchCV+KNeighborsClassifier"
    })

    cv_learner = ConvolutionalLearnerCV(units_per_layer=[height*width, 100, 10])

    n_train = 1000

    #my_cv_learner.fit(input_tensor[:n_train], output_tensor[:n_train])
    cv_learner.fit(input_tensor, output_tensor)
    #predictions = my_cv_learner.predict(input_tensor[n_train:])
    predictions = cv_learner.predict(input_tensor_test)
    #accuracy = accuracy_score(predictions, output_tensor[n_train:])
    accuracy = accuracy_score(predictions, output_tensor_test)
    accuracy_row.append({
        "dataset" : data_name,
        "accuracy" : accuracy,
        "model" : "MyCV+ConvolutionalMLP"
    })
    #print(f"Accuracy for {data_name} dataset: {accuracy}")

    gg = p9.ggplot()+\
        p9.theme(text=p9.element_text(size=30))+\
        p9.geom_line(
            p9.aes(
                x="epoch",
                y="loss",
                color="set_name"
            ),
            data = cv_learner.train_df)+\
        p9.geom_point(
            p9.aes(
                x="epoch",
                y="loss",
                color="set_name"
            ),
            data = cv_learner.min_df)+\
        p9.ggtitle(data_name)

    gg.save(f"C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW13/{data_name}.png", width=10, height=6)

accuracy_df = pd.DataFrame(accuracy_row)
print(accuracy_df)

acc = p9.ggplot()+\
    p9.theme(text=p9.element_text(size=10))+\
    p9.geom_point(
        p9.aes(
            x = "accuracy",
            y = "model"
        ),
        data = accuracy_df
    )+\
    p9.ggtitle("Test accuracy")+\
    p9.facet_grid('~dataset')

acc.save(f"C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW13/accuracy.png", width=10, height=6)


