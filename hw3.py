#import required libraries
import pandas as pd
import numpy as np
import matplotlib
import os
import urllib.request
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
import plotnine as p9

matplotlib.use("agg")

spam_data_path = "C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW3/spam_data.csv"
zip_data_path = "C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW3/zip_data.gz"

spam_data_url = "https://hastie.su.domains/ElemStatLearn/datasets/spam.data"
zip_data_url = "https://hastie.su.domains/ElemStatLearn/datasets/zip.train.gz"

#downloading zip.train data
if not os.path.exists(zip_data_path):
    urllib.request.urlretrieve(zip_data_url, zip_data_path)
    print("ZIP data file downloaded successfully")

else:
    print("ZIP data file already exist")

#downloading spam data
if not os.path.exists(spam_data_path):
    urllib.request.urlretrieve(spam_data_url, spam_data_path)
    print("Spam data file downloaded successfully")
    
else:
    print("Spam data dile already exist")


#read data
zip_df = pd.read_csv(zip_data_path, sep=" ", header=None).dropna(axis=1)
spam_df = pd.read_csv(spam_data_path, sep=" ", header=None)

zip_df.head(5)
spam_df.head(5)

spam_label_col = -1

#remove rows with non 0/1 labels in spam data
spam_df = spam_df[(spam_df[57]==0) | (spam_df[57]==1)]

spam_features = spam_df.iloc[:, :spam_label_col].to_numpy()
spam_labels = spam_df.iloc[:, spam_label_col].to_numpy()

zip_label_col_num = 0
zip_label_vec = zip_df.iloc[:, zip_label_col_num]
is_01 = zip_label_vec.isin([0, 1])
zip01_df = zip_df.loc[is_01, :]
is_label_col = zip_df.columns == zip_label_col_num
zip_features = zip01_df.iloc[:, ~is_label_col].to_numpy()
zip_labels = zip01_df.iloc[:, is_label_col].to_numpy()

data_dict = {
    "spam" : (spam_features, spam_labels),
    "zip" : (zip_features, zip_labels)
}

param_grid = {"n_neighbors": range(1, 21)}

knn = KNeighborsClassifier()

#initializing KFold
K = 3

classifiers = {
    'KNN': {
        'model': KNeighborsClassifier(),
        'param_grid': {'n_neighbors': list(range(1, 21))}
    },
    'LogisticRegression': {
        'model': make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, max_iter=1000)),
        'param_grid': {}
    }
}

pred_dict = {}

results = []

for dataset_name, (features, labels) in data_dict.items():
    print(f"Dataset: {dataset_name}")

    for classifier_name, classifier_info in classifiers.items():
        print(f"Classifier: {classifier_name}")

        kf = KFold(n_splits=K, shuffle=True, random_state=23)

        classifier = classifier_info["model"]

        fold_results = {
            "data_set": [],
            "fold_id": [],
            "algorithm": [],
            "test_accuracy_percent": []
        }

        for fold_id, (train_index, test_index) in enumerate(kf.split(features)):
            train_features, test_features = features[train_index], features[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]
            
            #get the most frequent class label in the training set
            most_frequent_label = pd.DataFrame(train_labels).value_counts().idxmax()
            
            #fit the classifier to the training data
            grid_search = GridSearchCV(estimator=classifier, param_grid=classifier_info['param_grid'], cv=5)
            grid_search.fit(train_features, train_labels)
            
            # Get the best hyperparameters
            best_params = grid_search.best_params_
            
            # Compute predictions on the test set
            if classifier_name == 'KNN':
                predictions = grid_search.best_estimator_.predict(test_features)
            elif classifier_name == 'LogisticRegression':
                predictions = grid_search.predict(test_features)
            elif classifier_name == 'featureless':
                predictions = np.full(len(test_labels), most_frequent_label)
            
            # Calculate test accuracy
            accuracy = accuracy_score(test_labels, predictions)
            
            # Store results in fold_results dictionary
            fold_results['data_set'].append(dataset_name)
            fold_results['fold_id'].append(fold_id)
            fold_results['algorithm'].append(classifier_name)
            #store accuracy after multiplying with 100 to convert it into percent
            fold_results['test_accuracy_percent'].append(accuracy * 100)
    
        results.append(pd.DataFrame(fold_results))

        pred_dict[f'{dataset_name}_{classifier_name}'] = predictions

# Combine results into a single DataFrame
results_df = pd.concat(results, ignore_index=True)

print(results_df)

#Visualize the data using plotnine
visualization_plot = p9.ggplot()+\
    p9.facet_grid(".~data_set")+\
    p9.geom_point(
        p9.aes(
            x="test_accuracy_percent",
            y="algorithm",
        ),
        data = results_df
    )


visualization_plot.save("C:/Users/np588/Desktop/MSCS/SEM2/Deep Learning/HW3/visualization.png")
