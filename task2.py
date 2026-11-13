
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42)


model = LinearRegression()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


lr_mae = mean_absolute_error(y_test, y_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
lr_r2 = r2_score(y_test, y_pred)


baseline_pred = np.full(len(y_test), np.mean(y_train))


baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_r2 = r2_score(y_test, baseline_pred)


results = pd.DataFrame({
    "Model": ["Baseline", "Linear Regression"],
    "MAE": [baseline_mae, lr_mae],
    "RMSE": [baseline_rmse, lr_rmse],
    "R2": [baseline_r2, lr_r2]})

print("\nREGRESSION METRICS COMPARISON")
print(results.round(2).to_string(index=False))


print("\nMODEL COMPARISON")

if lr_mae < baseline_mae and lr_rmse < baseline_rmse:
    print("Linear Regression performs better than the baseline.")
else:
    print("Linear Regression does not improve on both baseline metrics.")

if lr_r2 > baseline_r2:
    print("Linear Regression explains more variation in the target.")
else:
    print("Linear Regression does not improve R-squared.")



# Bonus task

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score


iris = load_iris()

X = iris.data


y_binary = (iris.target == 0).astype(int)


X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary)


model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)


y_prob = model.predict_proba(X_test)[:, 1]


auc = roc_auc_score(y_test, y_prob)

print("AUC Score:", round(auc, 2))


fpr, tpr, thresholds = roc_curve(y_test, y_prob)


plt.figure(figsize=(7, 5))

plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray",
         label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Setosa vs Others")

plt.legend()
plt.grid()

plt.show()
