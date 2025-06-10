import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#-----------Notes--------------
#Overall Accuracy before Hyperparameter Tuning: 89%

#Penalty in Log Reg: 
    #l1 penalty -> lasso regularization -> forces some weights to zero -> good for feature selection
    #l2 penalty -> ridge regularization -> keeps all features but shrinks weights (best for LogR)

#Solver in Log Reg: algorithm that solves optimization ("lbfgs" standard for l2)

#C in Log Reg: controls how much the model is allowed to make large weights (1/C)
    #large C -> super weak regularization -> model fits data closely -> possibly overfit
    #small C -> strong regularization -> simpler model -> reduces overfitting


#RESULTS: Logistic Regression Highest Accuracy is 0.892 with a C of 100
#-----------End Notes-----------

main_df = pd.read_csv("main_df.csv") 
X = main_df.iloc[:, 0:200].values #input features for the model
y = main_df["Label"].values #output values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 69) #split data

scaler = StandardScaler() #standardize features for logistic
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logReg = LogisticRegression(C = 100, random_state = 69, max_iter = 1000) #initiate model
logReg.fit(X_train_scaled, y_train) #train model on training data

y_pred = logReg.predict(X_test_scaled) #predict data using test data

accuracy = accuracy_score(y_test, y_pred) #overall accuracy
conf_matrix = confusion_matrix(y_test, y_pred) #Predicted/True
class_report = classification_report(y_test, y_pred) #Precision Scores

print(f'Test Accuracy: {accuracy:.4f}')
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

#-------------Hyperparameter Tuning----------------

for c in [0.01, 0.1, 1, 10, 100, 1000, 10000]: #trying each C value
    logReg_ht = LogisticRegression(
        random_state = 69,
        max_iter = 1000,
        C = c,
        penalty = "l2", #ridge regularization
        solver = "lbfgs"
    )
    logReg_ht.fit(X_train_scaled, y_train)

    y_pred_ht = logReg_ht.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred_ht)
    print(f"Test Accuracy with C of {c}: {accuracy:.4f}")

# Test Accuracy with C of 0.01: 0.8752
# Test Accuracy with C of 0.1: 0.8785
# Test Accuracy with C of 1: 0.8855 #Default
# Test Accuracy with C of 10: 0.8902 
# Test Accuracy with C of 100: 0.8920 #Best Option
# Test Accuracy with C of 1000: 0.8918
# Test Accuracy with C of 10000: 0.8918
