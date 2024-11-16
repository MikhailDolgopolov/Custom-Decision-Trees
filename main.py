# Импортируем необходимые библиотеки
import random

import graphviz
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.utils.estimator_checks import check_estimator

from AdaptiveDecisionTreeClassifier import AdaptiveDecisionTreeClassifier

# Загружаем набор данных Ирис
iris = load_iris()

# seed=random.randint(0,100)
seed=6

X = iris.data
y = iris.target


def test_and_render(data, output, sequence, depth=5, seed=0, prefix='', feature_names=None, class_names=None, colors=None):
    clf = AdaptiveDecisionTreeClassifier(sequence, max_depth=depth, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(data, output, test_size=0.3, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность модели s{seed}_[{'_'.join(map(str, sequence))}][depth{depth}]: {accuracy * 100:.2f}%")

    dot = clf.export_graphviz("graphs/iris.txt",
                              feature_names=feature_names,
                              class_names=class_names,
                              class_colors=colors,
                              impurity=False,
                              proportion=True,
                              precision=1)
    graph = graphviz.Source(dot)
    if prefix!='':
        prefix +='_'
    graph.render(
        f"graphs/{prefix}tree_s{seed}_[{'_'.join(map(str, sequence))}][depth{depth}]-{round(accuracy * 100, 1)}",
        format="png", cleanup=True)

# test_and_render(X, y, [1], 4, prefix='iris', class_names=iris.target_names)
depth=3
seq=[0]
clf = AdaptiveDecisionTreeClassifier(seq, max_depth=depth, feature_names=iris.feature_names, random_state=seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# print(f"Точность модели s{seed}_[{'_'.join(map(str, sequence))}][depth{depth}]: {accuracy * 100:.2f}%")
#"graphs/iris.txt"
dot = clf.export_graphviz(out_file=None,
                          feature_names=iris.feature_names,
                          class_names=iris.target_names,
                          label='all',
                          filled=False,
                          leaves_parallel=False,
                          impurity=True,
                          node_ids=True,
                          proportion=False,
                          rotate=False,
                          rounded=True,
                          special_characters=False,
                          precision=3)

graph = graphviz.Source(dot)

graph.render(
    f"graphs/iris_tree_s{seed}_[{'_'.join(map(str, seq))}][depth{depth}]-{round(accuracy * 100, 1)}",
    format="png", cleanup=True)

tree = DecisionTreeClassifier(max_depth=4, random_state=seed)

tree.fit(X, y)

dot = export_graphviz(tree,
                out_file=None,
                feature_names = None,
                class_names = None,
                label = 'all',
                filled = False,
                leaves_parallel = True,
                impurity = True,
                node_ids = True,
                proportion = False,
                rotate = False,
                rounded = True,
                special_characters = False,
                precision = 3,
                )
graph = graphviz.Source(dot)
graph.render(
        f"graphs/test_tree", format="png")