
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans


iris = load_iris()

X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y)


model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)

print("SUPERVISED LEARNING - LOGISTIC REGRESSION")
print("Accuracy:", round(accuracy, 2))
print("F1 (macro):", round(f1, 2))
print("Confusion Matrix:")
print(cm)


kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10)

clusters = kmeans.fit_predict(X)


cluster_sizes = np.bincount(clusters)

print("\nUNSUPERVISED LEARNING - KMEANS")
print("KMeans cluster sizes:", cluster_sizes.tolist())


print("\nCLUSTER COMPARISON WITH REAL SPECIES")

for cluster in range(3):
    indices = np.where(clusters == cluster)[0]
    actual_species = y[indices]

    counts = np.bincount(actual_species, minlength=3)

    print(f"Cluster {cluster}:")
    print("Setosa:", counts[0])
    print("Versicolor:", counts[1])
    print("Virginica:", counts[2])
