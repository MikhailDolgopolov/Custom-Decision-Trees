import re
from types import NoneType
from typing import List, Union, Optional
import graphviz
import pandas as pd
from graphviz import Source
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils._param_validation import InvalidParameterError

from helpers import generate_distinct_colors_hex


class AdaptiveDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(
            self,
            split_feature_order: Optional[List[Union[int, str]]] = None,
            handle_nans: bool = True,
            max_depth: Optional[int] = None,
            feature_names: Optional[List[str]] = None,
            **kwargs
    ) -> None:
        self.handle_nans = handle_nans
        if isinstance(feature_names, np.ndarray):
            feature_names = feature_names.tolist()
        self.feature_names = feature_names
        if max_depth is not None:
            if not isinstance(max_depth, int) or max_depth <= 0:
                raise InvalidParameterError(
                    f"The 'max_depth' parameter must be an int in the range [1, inf), got {max_depth} instead.")
        self.max_depth = max_depth
        self.split_feature_order = [] if split_feature_order is None else split_feature_order

        accepted_lists = (list, np.ndarray, pd.Index)
        if self.feature_names is not None and not isinstance(feature_names, (*accepted_lists, NoneType)):
            raise ValueError(f"Accepted types for feature_names are {', '.join(map(str, accepted_lists))}, not {type(self.feature_names)}.")

        if len(self.split_feature_order)>0:
            if all(isinstance(s, str) for s in self.split_feature_order):
                if self.feature_names is None:
                    raise ValueError("split_feature_order contains feature names, but no feature_names were provided.")
                if len(self.split_feature_order) > len(feature_names):
                    raise ValueError("Split sequence cannot be longer than the number of features you have.")
                missing_columns = [name for name in self.split_feature_order if
                                   isinstance(name, str) and name not in self.feature_names]
                if missing_columns:
                    raise ValueError(f"Columns {missing_columns} not found in the feature list you provided.")
                self.split_feature_order = [
                    list(feature_names).index(name) if isinstance(name, str) else name
                    for name in self.split_feature_order
                ]

            elif all(isinstance(i, int) for i in self.split_feature_order):
                if len(self.split_feature_order) != len(set(self.split_feature_order)):
                    raise ValueError("Provided split sequence contains duplicate indices.")
                if self.feature_names is not None and any(i >= len(self.feature_names) for i in self.split_feature_order):
                    raise ValueError("Provided split sequence contains invalid indices (out of bounds).")

            elif any(isinstance(k, (int, str)) for k in self.split_feature_order):
                raise InvalidParameterError("Incorrect format of split_feature_order. Use either exclusively indices, or exclusively feature names.")

        max_depth -= len(self.split_feature_order)
        super().__init__(max_depth=max_depth, **kwargs)
        self.__kwargs = kwargs
        if len(self.split_feature_order) > 0:
            self.split_index = self.split_feature_order[0]

    @property
    def tree_(self):
        raise ValueError("This estimator does not provide a full tree structure, because it is recursive.")

    def inpute(self, raw_X):
        self.imputer = SimpleImputer(strategy='median')
        if isinstance(raw_X, pd.DataFrame):
            if raw_X.isna().any().any():
                print("Handling NaN values with imputation.")
                return self.imputer.fit_transform(raw_X)
            return raw_X.to_numpy()
        else:
            X:np.ndarray = np.asarray(raw_X)
            if np.any(np.isnan(X)):
                print("Handling NaN values with imputation.")
                return self.imputer.fit_transform(X)
            return X

    def fit(self, X, y, **kwargs):
        self.depth = kwargs.get('depth', 0)

        X, y = check_X_y(X, y, accept_sparse=False)
        if self.depth == 0 and X.shape[0] == 0:
            raise ValueError("Training data cannot be empty!")
        parent_value:list|None = kwargs.get("parent_value", None)
        if len(y) == 0:
            if parent_value is not None:
                majority_class = np.argmax(parent_value)  # Most common class from parent's distribution
                self.leaf_model = lambda X: np.full((len(X),), majority_class)
                self.value = parent_value
            return self

        if self.handle_nans:
            X = self.inpute(X)

        self.X_, self.y_ = X, y
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError("Multi-output prediction is not supported by this estimator.")
        else:
            self.n_outputs_ = 1
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if len(self.split_feature_order) == 0 or self.depth >= len(self.split_feature_order):
            self.leaf_model = DecisionTreeClassifier(max_depth=self.max_depth, **self.__kwargs)
            self.leaf_model.fit(X, y)
            self.impurity = self._calculate_impurity(y)
            self.n_node_samples = len(y)
            self.value = np.bincount(y)
            self.split_index = self.leaf_model.tree_.feature[0]
            self.threshold = self.leaf_model.tree_.threshold[0]
            return self

        #Model to find the best threshold
        single_feature_model = DecisionTreeClassifier(
            criterion=self.criterion, max_depth=1, random_state=self.random_state)
        single_feature_model.fit(X[:, [self.split_index]], y)

        self.threshold = single_feature_model.tree_.threshold[0]

        left_mask = X[:, self.split_index] <= self.threshold
        right_mask = ~left_mask

        # Recursively train left and right subtrees
        self.left_tree = AdaptiveDecisionTreeClassifier(
            self.split_feature_order[1:], feature_names=self.feature_names, max_depth =self.max_depth - 1, **self.__kwargs)
        self.left_tree.fit(X[left_mask], y[left_mask],
                           depth=self.depth + 1, parent_value=getattr(self, 'value', None))

        self.right_tree = AdaptiveDecisionTreeClassifier(
            self.split_feature_order[1:], feature_names=self.feature_names, max_depth =self.max_depth - 1, **self.__kwargs)
        self.right_tree.fit(X[right_mask], y[right_mask],
                            depth=self.depth + 1, parent_value=getattr(self, 'value', None))

        self.impurity = self._calculate_impurity(y)
        self.n_node_samples = len(y)
        self.value = np.bincount(y)  # Class distribution at this node

        return self

    def _calculate_impurity(self, y):
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        impurity = 1 - np.sum(probabilities ** 2)
        return impurity

    def predict(self, X, check_input=True):
        if self.handle_nans:
            X = self.inpute(X)
        return np.array([self._predict_instance(instance, check_input) for instance in X])

    def _predict_instance(self, instance, check=True, depth=0):
        if getattr(self, 'leaf_model', None):
            if callable(self.leaf_model):
                prediction = self.leaf_model(np.array([instance]))[0]
                return prediction
            else:
                prediction = self.leaf_model.predict(instance.reshape(1, -1), check_input=check)[0]
                return prediction

        if depth >= len(self.split_feature_order):
            return super().predict(instance.reshape(1, -1))[0]

        split_index = self.split_feature_order[depth]
        if instance[split_index] <= self.threshold:
            return self.left_tree._predict_instance(instance, depth=depth + 1)
        else:
            return self.right_tree._predict_instance(instance, depth=depth + 1)

    def visualize(self, feature_names=None, class_names=None) -> Source:
        if feature_names is None:
            feature_names = self.feature_names
        if class_names is not None and len(class_names) < 2:
            class_names = None
        if not getattr(self, 'leaf_model', None) and \
                not getattr(self, 'right_tree', None) and \
                not getattr(self, 'left_tree', None):
            raise RuntimeError("The Classifier is not fitted!")
        fill = '' if class_names is None else ', filled'
        dot = 'digraph Tree {\n' + \
              f'node [shape=box, style="rounded{fill}", color="black", fontname="helvetica"] ;\n' + \
              'edge [fontname="helvetica"] ;\n'

        colorMap = None
        if class_names is not None and len(class_names) > 1:
            new_colors = generate_distinct_colors_hex(len(class_names))
            colorMap = {class_names[i]: new_colors[i] for i in range(len(class_names))}
        dot_str, node = self.build_dot(feature_names, class_names, nodes=0, color_map=colorMap)
        dot += dot_str
        dot += '\n}'

        return graphviz.Source(dot, filename=None)

    def build_dot(self, feature_names, class_names, nodes=0, color_map=None):
        dot_str = ""
        try:
            feature_name = feature_names[self.split_index]
        except:
            feature_name = f"Feature {self.split_index}"
        threshold = self.threshold
        impurity = self.impurity
        samples = self.n_node_samples

        class_index = np.argmax(self.value)  # Get the majority class index
        try:
            class_name = class_names[class_index]
        except:
            class_name = f"Class {class_index}"

        label = (f'<{feature_name} &le; {threshold:.2f}<br/>gini = {impurity:.3f}<br/>'
                 f'samples = {samples}<br/> class="{class_name}">')

        my_fill = f'"{color_map[class_name]}"' if color_map else '"#fdfcff"'
        dot_str += f'{nodes} [label={label}, fillcolor={my_fill}] ;\n'
        nodes+=1
        new_features = feature_names
        new_quantity = 1
        if getattr(self, 'left_tree', None):
            # Recursively build the left tree
            left_dot_str, new_quantity = self.left_tree.build_dot(new_features, class_names, nodes=nodes,
                                                               color_map=color_map)

            dot_str += f'{nodes-1} -> {nodes};\n'
            dot_str += left_dot_str
        if getattr(self, 'right_tree', None):
            # Recursively build the right tree
            right_child = new_quantity+1
            right_dot_str, new_quantity = self.right_tree.build_dot(new_features, class_names, nodes=right_child,
                                                                 color_map=color_map)
            dot_str += right_dot_str
            dot_str += f'{nodes-1} -> {right_child};\n'
            nodes = new_quantity

        if getattr(self, 'leaf_model', None):
            # Visualize the entire leaf model
            new_dot, new_quantity = self.visualize_leaf_tree(nodes, feature_names, class_names, color_map)
            dot_str += new_dot
        return dot_str, new_quantity

    def visualize_leaf_tree(self, nodes, feature_names, class_names, class_colors):
        leaf_dot = export_graphviz(
            self.leaf_model,
            out_file=None,  # Return as a string
            feature_names=feature_names,
            class_names=class_names,
            rounded=True,
            special_characters=True
        )

        if class_colors:
            for class_name, color in class_colors.items():
                leaf_dot = re.sub(
                    f'class *= *{class_name}>',
                    f'class = "{class_name}"> fillcolor="{color}"',
                    leaf_dot
                )

        def shift_index(match):
            index = int(match.group(1))
            return f" {index + nodes-1}"

        filtered_lines = []
        for line in leaf_dot.splitlines():
            # Keep only lines that define a node or edge (node index or node -> connection)
            if '->' in line or re.match(r'\d+ +(?:\[.*\])*', line):
                shifted_line = re.sub(r'(\b\d+\b)(?=\s*[\[;]| ->)', shift_index, line).strip()
                filtered_lines.append(shifted_line)
        filtered_dot = "\n".join(filtered_lines)

        size = getattr(self.leaf_model, "tree_.node_count", 1)
        return filtered_dot, nodes+size
