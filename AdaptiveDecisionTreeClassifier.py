import re
import warnings
from types import NoneType
from typing import List, Union, Optional
import graphviz
import pandas as pd
from graphviz import Source
from sklearn.impute import SimpleImputer
from sklearn.metrics import mutual_info_score
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
        if max_depth is not None:
            if not isinstance(max_depth, int) or max_depth <= 0:
                raise InvalidParameterError(
                    f"The 'max_depth' parameter must be an int in the range [1, inf), got {max_depth} instead.")
        self.overall_max_depth = max_depth
        if isinstance(max_depth, int):
            max_depth -= len(self.split_feature_order)
        super().__init__(max_depth=max_depth, **kwargs)
        self.max_depth = max_depth

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
                print("Handling NaN values with median imputation.")
                return self.imputer.fit_transform(raw_X)
            return raw_X.to_numpy()
        else:
            X:np.ndarray = np.asarray(raw_X)
            if np.any(np.isnan(X)):
                print("Handling NaN values with median imputation.")
                return self.imputer.fit_transform(X)
            return X

    def fit(self, X, y, fit_depth=0, bad_split_error_threshold=0, **kwargs):
        if fit_depth == 0 and X.shape[0] == 0:
            raise ValueError("Training data cannot be empty!")
        else:
            X, y = check_X_y(X, y, accept_sparse=False)
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

        if len(self.split_feature_order) == 0 or isinstance(fit_depth, int) and fit_depth >= len(self.split_feature_order):
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
            criterion="gini", max_depth=1, random_state=self.random_state)
        single_feature_model.fit(X[:, [self.split_index]], y)

        self.threshold = single_feature_model.tree_.threshold[0]

        def compute_impurity(mask):
            if mask.sum() == 0:
                return 0  # Avoid division by zero for empty nodes
            prob = np.bincount(y[mask]) / mask.sum()

            return 1 - np.sum(prob ** 2)  # Gini impurity

        left_mask = X[:, self.split_index] <= self.threshold
        right_mask = ~left_mask

        n_left, n_right = left_mask.sum(), right_mask.sum()
        left_impurity = compute_impurity(left_mask)
        right_impurity = compute_impurity(right_mask)
        selected_split_impurity = (n_left * left_impurity + n_right * right_impurity) / len(y)
        selected_split_gain = mutual_info_score(y, X[:, self.split_index] <= self.threshold)

        if bad_split_error_threshold>0:
            if self.criterion not in ['entropy', 'gini']:
                raise NotImplementedError(f"Split evaluation is not supported for criterion '{self.criterion}'.")
            # Find the best split across all features
            best_split_impurity = 1  # Initialize with maximum impurity
            best_split_gain = 0
            best_feature_gini, best_feature_entropy = None, None
            for feature_idx in range(X.shape[1]):
                temp_model = DecisionTreeClassifier(
                    criterion=self.criterion, max_depth=1, random_state=self.random_state)
                temp_model.fit(X[:, [feature_idx]], y)
                temp_threshold = temp_model.tree_.threshold[0]

                temp_gain = mutual_info_score(y, X[:, feature_idx] <= temp_threshold)
                if temp_gain > best_split_gain:
                    best_split_gain = temp_gain
                    best_feature_entropy = feature_idx

                temp_left_mask = X[:, feature_idx] <= temp_threshold
                temp_right_mask = ~temp_left_mask

                temp_n_left, temp_n_right = temp_left_mask.sum(), temp_right_mask.sum()
                temp_left_impurity = compute_impurity(temp_left_mask)
                temp_right_impurity = compute_impurity(temp_right_mask)
                temp_split_impurity = (temp_n_left * temp_left_impurity + temp_n_right * temp_right_impurity) / len(y)

                if temp_split_impurity<best_split_impurity:
                    best_split_impurity = temp_split_impurity
                    best_feature_gini = feature_idx
                best_split_impurity = min(temp_split_impurity, best_split_impurity)

            split = self.feature_names[self.split_index] if getattr(self, "feature_names", False) else f"index {self.split_index}"
            if self.criterion=='gini':
                best_split = self.feature_names[best_feature_gini] if getattr(self, "feature_names", False) else f"index {best_feature_gini}"
                if selected_split_impurity > best_split_impurity + bad_split_error_threshold:
                    warnings.warn(
                        f"The split at {split} is worse than the best possible split, at {best_split}. \n"
                        f"Selected impurity: {selected_split_impurity:.4f}, Best impurity: {best_split_impurity:.4f}. \n"
                        f"Consider revising your split feature order."
                    )
            if self.criterion == 'entropy':
                best_split = self.feature_names[best_feature_entropy] if getattr(self, "feature_names",
                                                                              False) else f"index {best_feature_entropy}"
                if selected_split_gain + bad_split_error_threshold < best_split_gain:
                    warnings.warn(
                        f"The split at {split} is worse than the best possible split, at {best_split}. \n"
                        f"Selected split reduces entropy by: {selected_split_gain:.4f}, Best one: {best_split_gain:.4f}. \n"
                        f"Consider revising your split feature order."
                    )

        next_init_depth = self.overall_max_depth-1 if isinstance(self.overall_max_depth, int) else None
        # Recursively train left and right subtrees
        self.left_tree = AdaptiveDecisionTreeClassifier(
            self.split_feature_order[1:], feature_names=self.feature_names, max_depth = next_init_depth, **self.__kwargs)
        self.left_tree.fit(X[left_mask], y[left_mask], bad_split_error_threshold=bad_split_error_threshold,
                           fit_depth=fit_depth + 1, parent_value=getattr(self, 'value', None))

        self.right_tree = AdaptiveDecisionTreeClassifier(
            self.split_feature_order[1:], feature_names=self.feature_names, max_depth = next_init_depth, **self.__kwargs)
        self.right_tree.fit(X[right_mask], y[right_mask], bad_split_error_threshold=bad_split_error_threshold,
                            fit_depth=fit_depth + 1, parent_value=getattr(self, 'value', None))

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

    def visualize(self, feature_names=None, class_names=None, class_colors=None) -> Source:
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
        if class_names is not None and class_colors is not None:
            if self.n_classes_ != len(class_colors):
                raise ValueError(f"Wrong number of colors provided. Should be: {self.n_classes_}, got: {len(class_colors)}.")
        colorMap = None
        if class_names is not None and len(class_names) > 1:
            if class_colors is None:
                class_colors = generate_distinct_colors_hex(len(class_names))
            colorMap = {class_names[i]: class_colors[i] for i in range(len(class_names))}
        dot_str, node = self.build_dot(feature_names, class_names, nodes=0, color_map=colorMap)
        dot += dot_str
        dot += '\n}'

        return graphviz.Source(dot, filename=None)

    def build_dot(self, feature_names, class_names, nodes=0, color_map=None):
        dot_str = ""
        try:
            feature_name = feature_names[self.split_index]
        except:
            feature_name = f"x<SUB>{self.split_index}</SUB>"
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

        size = 1
        if hasattr(self.leaf_model, "tree_"):
            size = self.leaf_model.tree_.node_count
        return filtered_dot, nodes+size
