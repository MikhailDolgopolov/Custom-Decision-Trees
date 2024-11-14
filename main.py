# Импортируем необходимые библиотеки
import random

import graphviz
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.estimator_checks import check_estimator

from RecursiveCustomDecisionTreeClassifier import RecursiveCustomDecisionTreeClassifier
from helpers import save_dot_to_file

# Загружаем набор данных Ирис
iris = load_iris()

# seed=random.randint(0,100)
seed=5
# Разделяем данные на признаки и целевые метки
X = iris.data  # признаки
y = iris.target  # целевые метки

# Разделяем данные на обучающую и тестовую выборки (70% на обучение, 30% на тестирование)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# Создаем модель дерева решений
seq=[0, 3]
clf = RecursiveCustomDecisionTreeClassifier(random_state=seed,split_sequence=seq, max_depth=1)

# Обучаем модель на обучающих данных
clf.fit(X_train, y_train)

# Предсказываем целевые метки на тестовой выборке
y_pred = clf.predict(X_test)
#
# # Оцениваем точность модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy * 100:.2f}%')

print(iris.feature_names)
#iris.feature_names, iris.target_names
graph = clf.visualize(iris.feature_names, iris.target_names)

# Render the tree (display it)
graph.render(
    f"iris_tree_s{seed}_[{'_'.join(map(str, seq))}]-{round(accuracy * 100)}",
    format="png", cleanup=True)
