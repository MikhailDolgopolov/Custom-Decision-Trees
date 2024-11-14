from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Source
import numpy as np


class CustomDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self,  split_index, **kwargs):
        self.first_feature_index = split_index
        super().__init__(**kwargs)
        self.single_feature_model = None


    def fit(self, X, y, sample_weight=None, check_input=True):
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))  # Количество уникальных классов
        self.classes_ = np.unique(y)  # Массив классов
        # Временно создаем модель с одним параметром для вычисления оптимального порога
        self.single_feature_model = DecisionTreeClassifier(
            criterion=self.criterion, max_depth=1, random_state=self.random_state
        )
        self.single_feature_model.fit(X[:, [self.first_feature_index]], y)

        # Определяем оптимальный порог для разбиения по этому параметру
        self.threshold = self.single_feature_model.tree_.threshold[0]

        # Разбиваем данные на основе этого порога
        left_mask = X[:, self.first_feature_index] <= self.threshold
        right_mask = ~left_mask

        X_left = np.delete(X[left_mask], self.first_feature_index, axis=1)
        X_right = np.delete(X[right_mask], self.first_feature_index, axis=1)

        # Обучаем поддеревья для каждой ветви, указав разное начальное подмножество данных
        self.left_tree = DecisionTreeClassifier(criterion=self.criterion, random_state=self.random_state)
        self.right_tree = DecisionTreeClassifier(criterion=self.criterion, random_state=self.random_state)

        # Строим поддеревья для левой и правой ветвей
        self.left_tree.fit(X_left, y[left_mask])
        self.right_tree.fit(X_right, y[right_mask])

    def predict(self, X, check_input=True):
        # Предсказываем с учетом первого разбиения
        predictions = np.zeros(X.shape[0], dtype=int)

        # Разделяем данные на основе порога
        left_mask = X[:, self.first_feature_index] <= self.threshold
        right_mask = ~left_mask

        # Убираем первый признак и делаем предсказания для каждой группы данных
        X_left = np.delete(X[left_mask], self.first_feature_index, axis=1)
        X_right = np.delete(X[right_mask], self.first_feature_index, axis=1)

        # Используем поддеревья для предсказаний в каждой ветви
        predictions[left_mask] = self.left_tree.predict(X_left, check_input=check_input)
        predictions[right_mask] = self.right_tree.predict(X_right, check_input=check_input)

        return predictions

    def export_graph(self, feature_names=None, class_names=None, filled=True):
        # Выводим корневой узел с основным разбиением
        threshold = self.threshold  # убедитесь, что self.threshold определён
        dot_data = f"""
                digraph Tree {{
                    node [shape=box, style="filled", color="lightgrey"];
                    root [label="Split on feature {feature_names[self.first_feature_index] if feature_names else self.first_feature_index} <= {threshold}", fillcolor="#e58139"];
                    root -> left_node;
                    root -> right_node;
                """

        # Экспортируем левое поддерево и добавляем его узлы в общий граф
        left_dot_data = export_graphviz(
            self.left_tree, out_file=None,
            feature_names=[name for i, name in enumerate(feature_names) if
                           i != self.first_feature_index] if feature_names else None,
            class_names=class_names,
            filled=filled
        )

        # Экспортируем правое поддерево и добавляем его узлы в общий граф
        right_dot_data = export_graphviz(
            self.right_tree, out_file=None,
            feature_names=[name for i, name in enumerate(feature_names) if
                           i != self.first_feature_index] if feature_names else None,
            class_names=class_names,
            filled=filled
        )

        # Добавляем левое и правое поддеревья как узлы
        dot_data += left_dot_data.replace("digraph Tree {", "").replace("}", "").replace("node [", "left_node [")
        dot_data += right_dot_data.replace("digraph Tree {", "").replace("}", "").replace("node [", "right_node [")
        dot_data += "\n}"

        # Отображаем дерево с помощью Graphviz
        graph = Source(dot_data)
        return graph