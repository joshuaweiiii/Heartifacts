import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


main_df = pd.read_csv("main_df.csv")
X = main_df.iloc[:, 0:200].values
y = main_df["Label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 69)

