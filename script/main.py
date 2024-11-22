from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import mlflow
from mlflow import MlflowClient

breast_cancer_data = datasets.load_breast_cancer(as_frame=True)

print(breast_cancer_data["data"].head())

models = {}

# Logistic Regression
from sklearn.linear_model import LogisticRegression
models['Logistic Regression'] = LogisticRegression()

# Support Vector Machines
from sklearn.svm import LinearSVC
models['Support Vector Machines'] = LinearSVC()

# Decision Trees
from sklearn.tree import DecisionTreeClassifier
models['Decision Trees'] = DecisionTreeClassifier()

# Random Forest
from sklearn.ensemble import RandomForestClassifier
models['Random Forest'] = RandomForestClassifier()

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
models['Naive Bayes'] = GaussianNB()

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
models['K-Nearest Neighbor'] = KNeighborsClassifier()

X_train, X_test, y_train, y_test = train_test_split(
    breast_cancer_data.data,
    breast_cancer_data.target,
    test_size = 0.2,
    random_state=42
)

ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)

ss_test = StandardScaler()
X_test = ss_test.fit_transform(X_test)

# Impression des shapes de données
print("X_train :\t", X_train.shape)
print("y_train :\t", y_train.shape)
print("X_test :\t", X_test.shape)
print("y_test :\t", y_test.shape)

client = MlflowClient()

mlflow.set_tracking_uri("http://localhost:5000")

# Modèles
mlflow.set_experiment("Modèles sans le StdScaler")
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Random Forest"):
  clf = models["Random Forest"]
  clf.fit(X_train, y_train)

with mlflow.start_run(run_name="Régression Logistique"):
  clf2 = models["Logistic Regression"]
  clf2.fit(X_train, y_train)

with mlflow.start_run(run_name="SVM"):
  clf3 = models["Support Vector Machines"]
  clf3.fit(X_train, y_train)

with mlflow.start_run(run_name="Arbre de décision"):
  clf4 = models["Decision Trees"]
  clf4.fit(X_train, y_train)

with mlflow.start_run(run_name="Naive Bayes"):
  clf5 = models["Naive Bayes"]
  clf5.fit(X_train, y_train)

with mlflow.start_run(run_name="K-Nearest Neighbor"):
  clf6 = models["K-Nearest Neighbor"]
  clf6.fit(X_train, y_train)

client.create_registered_model("Random Forest")
client.create_registered_model("Régression Logistique")
client.create_registered_model("SVM")
client.create_registered_model("Arbre de décision")
client.create_registered_model("Naive Bayes")
client.create_registered_model("K-Nearest Neighbor")
