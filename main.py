# Импортируем необходимые библиотеки
import random

import graphviz
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.estimator_checks import check_estimator

from RecursiveCustomDecisionTreeClassifier import RecursiveCustomDecisionTreeClassifier
# check_estimator(RecursiveCustomDecisionTreeClassifier())
# Загружаем набор данных Ирис
iris = load_iris()

# seed=random.randint(0,100)
seed=5

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

seq=[3]
clf = RecursiveCustomDecisionTreeClassifier(random_state=seed,split_sequence=seq, max_depth=3)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy * 100:.2f}%')

graph = clf.visualize(iris.feature_names, iris.target_names)

graph.render(
    f"iris_tree_s{seed}_[{'_'.join(map(str, seq))}]-{round(accuracy * 100, 1)}",
    format="png", cleanup=True)
