import numpy as np
import torch
import math
import matplotlib
import pandas as pd
import torchvision
import plotnine as p9
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

matplotlib.use('agg')

data_info_dict = {
    "zip" : ("c:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW11/zip.test.gz", 0),
    "spam" : ("C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW5/spam_data.csv", -1)
}
data_dict = {}

for data_name, (file_name, label_col_num) in data_info_dict.items():
    data_df = pd.read_csv(file_name, sep = " ", header = None)
    #print(f"{data_name}\n{data_df}")
    data_label_vec = data_df.iloc[:, label_col_num]
    is_label_col = data_df.columns == label_col_num
    data_features = data_df.iloc[:, ~is_label_col]
    data_labels = data_df.iloc[:, is_label_col]
    print("%s %s"%(data_name, data_features.shape))

{data_name:features.shape for data_name, (features, labels) in data_dict.items()}

class TorchModel(torch.nn.Module):
    def __init__(self, units_per_layer):
        super(TorchModel, self).__init__()
        seq_args = []
        second_to_last = len(units_per_layer)-1
        for layer_i in range (second_to_last):
            next_i = layer_i + 1
            layer_units = units_per_layer[layer_i]
            next_units = units_per_layer[next_i]
            seq_args.append(torch.nn.Linear(layer_units, next_units))
            if layer_i < second_to_last-1:
                seq_args.append(torch.nn.ReLU())
        self.stack = torch.nn.Sequential(*seq_args)
    def forward(self, features):
        return self.stack(features)

class CSV(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __getitem__(self, item):
        return self.features[item, :], self.labels[item]
    def __len__(self):
        return len(self.labels)

class TorchLearner:
    def __init__(
            self, units_per_layer, step_size = 0.1,
            batch_size = 20, max_epochs = 100):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.model = TorchModel(units_per_layer)
        self.loss_fun = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=step_size)
    def fit(self, split_data_dict):
        ds = CSV(
            split_data_dict["subtrain"]["X"],
            split_data_dict["subtrain"]["y"])
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=True)
        train_df_list = []
        for epoch_number in range(self.max_epochs):
            #print(epoch_number)
            for batch_features,  batch_labels in dl:
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

class TorchLearnerCV:
    def __init__(self, n_folds = 3, units_per_layer = [data_ncol, 1]):
        self.units_per_layer = units_per_layer
        self.n_folds = n_folds
    def fit(self, train_features, train_labels):
        train_nrow, train_ncol = train_features.shape
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
            learner = TorchLearner(self.units_per_layer)
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
        self.final_learner = TorchLearner(
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
            is_split = {
                "subtrain" : fold_id_vec != validation_fold,
                "validation" : fold_id_vec == validation_fold
            }
            split_data_dict = {}
            for set_name, is_set in is_split.items():  
                split_data_dict[set_name] = (
                    X[is_set],
                    y[is_set].reshape(-1,1)
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
        self.best_param_dict = self.param_grid[best_index]
        self.fit_one(self.best_param_dict, X, y)
    def predict(self, X):
        self.estimator.predict(X)

for data_name, (data_features, data_labels) in data_dict.items():
    input_tensor, output_tensor = data_dict[data_name]
    data_nrow, data_ncol = input_tensor.shape
    hyper_params = []
    for n_layers in range(5):
        hyper_params.append({
            "units_per_layer" : [
                data_ncol
            ]+[10 for layer_num in range(n_layers)]+[n_classes]
        })

    my_cv_learner = MyCV(
        estimator = TorchLearnerCV(),
        param_grid = hyper_params,
        cv = 2)

    n_train = 1000

    my_cv_learner.fit(input_tensor[:n_train], output_tensor[:n_train])

    gg = p9.ggplot()+\
        p9.theme(text=p9.element_text(size=30))+\
        p9.geom_line(
            p9.aes(
                x="epoch",
                y="loss",
                color="set_name"
            ),
            data = my_cv_learner.estimator.train_df)+\
        p9.geom_point(
            p9.aes(
                x="epoch",
                y="loss",
                color="set_name"
            ),
            data = my_cv_learner.estimator.min_df)+\
        p9.ggtitle(data_name)

    gg.save(f"C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW11/{data_name}.png", width=10, height=6)        
