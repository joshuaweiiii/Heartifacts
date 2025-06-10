import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#-----------Notes--------------
#Kernel = defines the shape of the decision boundary
#C = how tolerant model should be of small errors
    #Large C = low regularization, risk of overfitting
    #Small C = high regularization, risk of underfitting
#Gamma = how far the influence of a single training point extends
    #gamma = "scale" -> 1 / (number of features * variance of X) (automatically adopts gamma to your data)

#Model Accuracy before Hyperparameter Tuning: 98.725%
#Model Accuracy after Hyperparameter Tuning: 99.225%

# Test Accuracy: 0.99225
# Confusion Matrix:
# [[1986   13]
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
y = main_df['Label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 69)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(
    kernel = 'rbf',      
    C = 10,             
    gamma = 'scale',    
    random_state = 69
)
svm.fit(X_train_scaled, y_train)

y_pred = svm.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Test Accuracy: {accuracy}')
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

#-----------Hyperparameter Tuning-----------

for c in [1, 10, 100, 1000]:
    svm_ht = SVC(
        kernel = "rbf",
        C = c,
        gamma = "scale",
        random_state = 69
    )

    svm_ht.fit(X_train_scaled, y_train)

    y_pred_ht = svm_ht.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred_ht)
    print(f'Accuracy with C of {c}: {accuracy}')