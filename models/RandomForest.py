import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#-----------Notes--------------
#n_jobs = parallelism control parameter, using -1 tells computer to use all CPU power possible
#Started off with 10 estimators, bore 0.987 accuracy
    #After tuning, 1000 estimators bore 0.9905 accuracy
#-----------End Notes-----------

main_df = pd.read_csv("main_df.csv")

X = main_df.iloc[:, 0:200].values
y = main_df["Label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 69)

randomForest = RandomForestClassifier(
    n_estimators = 10,
    max_depth = None,
    random_state = 69,
    n_jobs = -1 #uses all available CPU cores (fastest option)
)

randomForest.fit(X_train, y_train)

y_pred = randomForest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

#-------------Hyperparameter Tuning----------------

for estimator in [10, 100, 1000]: #trying different estimator counts
    randomForest_ht = RandomForestClassifier(
        n_estimators = estimator,
        max_depth = None,
        random_state = 69,
        n_jobs = -1
    )
    randomForest_ht.fit(X_train, y_train)
    y_pred_ht = randomForest_ht.predict(X_test)

    accuracy_ht =  accuracy_score(y_test, y_pred_ht)

    print(f'Accuracy with {estimator} estimators: {accuracy_ht}')