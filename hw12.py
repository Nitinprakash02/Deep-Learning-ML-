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

ds = torchvision.datasets.MNIST(
    root = "C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW11",
    download = True,
    transform = torchvision.transforms.ToTensor(),
    train = False)

dl = torch.utils.data.DataLoader(ds, batch_size = len(ds), shuffle = False)

for mnist_features, mnist_labels in dl:
    pass

mnist_features.flatten(start_dim = 1).numpy()
mnist_labels.numpy()

data_info_dict = {
    "zip" : ("c:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW11/zip.test.gz", 0),
    "spam" : ("C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW5/spam_data.csv", 57)
}
data_dict = {"mnist" : (mnist_features.flatten(start_dim=1), mnist_labels)}

for data_name, (file_name, label_col_num) in data_info_dict.items():
    data_df = pd.read_csv(file_name, sep = " ", header = None)
    #print(f"{data_name}\n{data_df}")
    data_label_vec = data_df.iloc[:, label_col_num]
    is_label_col = data_df.columns == label_col_num
    data_features = data_df.iloc[:, ~is_label_col]
    data_labels = data_df.iloc[:, is_label_col]
    if data_name == "spam":
        X_mean = data_features.mean()
        X_std = data_features.std()
        data_features = (data_features - X_mean) / X_std
    print("%s %s"%(data_name, data_features.shape))
    data_dict[data_name] = (
        torch.from_numpy(data_features.to_numpy()).float(),
        torch.from_numpy(data_labels.to_numpy()).flatten()
    )

{data_name:features.shape for data_name, (features, labels) in data_dict.items()}

input_tensor, output_tensor = data_dict["mnist"]
data_nrow, data_ncol = data_dict["mnist"][0].shape

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

n_classes = 10

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
            self, units_per_layer,
            batch_size = 20, max_epochs = 100):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.model = TorchModel(units_per_layer)
        self.loss_fun = torch.nn.CrossEntropyLoss()
        self.initial_step_size=0.1
        self.end_step_size=0.001
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
    data_nrow, data_ncol = input_tensor.shape
    hyper_params = []

    #Logistic regression
    clf = LogisticRegressionCV(cv=3, random_state=0)
    clf.fit(input_tensor, output_tensor)
    lr_predictions = clf.predict(input_tensor_test)
    lr_accuracy = accuracy_score(lr_predictions, output_tensor_test)

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

    grid_search = grid.fit(input_tensor, output_tensor)

    knn = KNeighborsClassifier(n_neighbors = grid_search.best_params_['n_neighbors'])
        
    knn.fit(input_tensor, output_tensor)
    knn_predictions = knn.predict(input_tensor_test)
    knn_accuracy = accuracy_score(knn_predictions, output_tensor_test)

    accuracy_row.append({
        "dataset" : data_name,
        "accuracy" : knn_accuracy,
        "model" : "GridSearchCV+KNeighborsClassifier"
    })

    cv_learner = TorchLearnerCV(units_per_layer=[data_ncol, 100, 10])
    #my_cv_learner = MyCV(cv_learner)
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
        "model" : "MyCV+OptimizerMLP(decreasing_lr & momentum)"
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

    gg.save(f"C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW12/{data_name}.png", width=10, height=6)

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
    p9.facet_grid("~dataset")
    #p9.scale_x_continuous(limits=(10,100), expand=(0, 0))

acc.save(f"C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW12/accuracy.png", width=10, height=6)
