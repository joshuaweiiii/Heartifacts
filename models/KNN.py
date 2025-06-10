import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#-----------Notes--------------
#Overall Accuracy before Hyperparameter Tuning and no Scaling: 87%
#Overall Accuracy before Hyperparameter Tuning and Scaling: 98.9%
#Overall Accuracy after Hyperparameter Tuning: 99.2%

# Test Accuracy: 0.9920
# Confusion Matrix:
# [[1985   14]
#  [  18 1983]]
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.99      0.99      0.99      1999
#            1       0.99      0.99      0.99      2001

#     accuracy                           0.99      4000
#    macro avg       0.99      0.99      0.99      4000
# weighted avg       0.99      0.99      0.99      4000

#-----------End Notes-----------

main_df = pd.read_csv("main_df.csv")

X = main_df.iloc[:, 0:200].values
y = main_df["Label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 69)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(
    n_neighbors = 1, #standard
    n_jobs = -1
)

knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred) #overall accuracy
conf_matrix = confusion_matrix(y_test, y_pred) #Predicted/True
class_report = classification_report(y_test, y_pred) #Precision Scores

print(f'Test Accuracy: {accuracy:.4f}')
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

#-------------Hyperparameter Tuning----------------

for k in [1, 2, 3, 4, 5, 10, 100, 1000, 10000]:
    knn_ht = KNeighborsClassifier(
        n_neighbors = k,
        n_jobs = -1
    )
    knn_ht.fit(X_train_scaled, y_train)

    y_pred_ht = knn_ht.predict(X_test_scaled)

    accuracy_ht = accuracy_score(y_test, y_pred_ht)
    
    print(f'Accuracy with K of {k}: {accuracy_ht}')