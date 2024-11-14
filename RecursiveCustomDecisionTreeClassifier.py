import re

import graphviz
from graphviz import Digraph, Source
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import numpy as np
from sklearn.utils import check_X_y

from helpers import generate_distinct_colors_hex


class RecursiveCustomDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, split_sequence=None, **kwargs):
        if split_sequence is None:
            split_sequence = []
        self.split_sequence = split_sequence  # List of feature indices for splits
        super().__init__(**kwargs)
        self.__kwargs = kwargs
        if len(split_sequence) > 0: self.split_index = split_sequence[0]

    @property
    def tree_(self):
        if getattr(self, 'leaf_model', None):
            return self.leaf_model.tree_
        else:
            return getattr(self.left_tree, 'tree_', None)

    def fit(self, X, y, **kwargs):
        X, y = check_X_y(X, y, accept_sparse=False)

        # Check if the input data is empty
        if X.shape[0] == 0 or len(y) == 0:
            raise ValueError("Training data is empty.")
        self.depth = kwargs.get('depth', 0)
        self.X_, self.y_ = X, y
        if len(y.shape) > 1 and y.shape[1] > 1:
            # If y is multi-output, set n_outputs_ to the number of columns in y
            self.n_outputs_ = y.shape[1]
        else:
            # Otherwise, it's a single-output problem
            self.n_outputs_ = 1
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        # Check if we've exhausted the split sequence
        if len(self.split_sequence) == 0:
            # At leaf, train a regular tree on remaining data
            self.leaf_model = DecisionTreeClassifier(**self.__kwargs)
            self.leaf_model.fit(X, y)
            self.impurity = self._calculate_impurity(y)  # Calculate impurity for leaf node
            self.n_node_samples = len(y)  # Number of samples at the leaf
            self.value = np.bincount(y)  # Class distribution at the leaf
            self.split_index = self.leaf_model.tree_.feature[0]
            self.threshold = self.leaf_model.tree_.threshold[0]
            return self

        # Train a single-feature model to find the best threshold
        single_feature_model = DecisionTreeClassifier(
            criterion=self.criterion, max_depth=1, random_state=self.random_state)
        single_feature_model.fit(X[:, [self.split_index]], y)

        # Retrieve the threshold for this split
        self.threshold = single_feature_model.tree_.threshold[0]

        # Split data into left and right based on the threshold
        left_mask = X[:, self.split_index] <= self.threshold
        right_mask = ~left_mask

        # X_left = np.delete(X[left_mask], self.split_index, axis=1)
        # X_right = np.delete(X[right_mask], self.split_index, axis=1)

        X_left = X[left_mask]
        X_right = X[right_mask]

        # Recursively train left and right subtrees
        self.left_tree = RecursiveCustomDecisionTreeClassifier(
            self.split_sequence[1:], **self.__kwargs)
        self.left_tree.fit(X_left, y[left_mask], depth=self.depth + 1)

        self.right_tree = RecursiveCustomDecisionTreeClassifier(
            self.split_sequence[1:], **self.__kwargs)
        self.right_tree.fit(X_right, y[right_mask], depth=self.depth + 1)

        self.impurity = self._calculate_impurity(y)  # Impurity for internal node
        self.n_node_samples = len(y)  # Number of samples at this node
        self.value = np.bincount(y)  # Class distribution at this node

        return self

    def _calculate_impurity(self, y):
        # Gini impurity calculation
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        impurity = 1 - np.sum(probabilities ** 2)
        return impurity

    def predict(self, X, check_input=True):
        # Iterate through each sample and predict recursively
        return np.array([self._predict_instance(instance, check_input) for instance in X])

    def _predict_instance(self, instance, check=True, depth=0):
        if getattr(self, 'leaf_model', None):
            # Use leaf model if this is a leaf
            return self.leaf_model.predict(instance.reshape(1, -1), check_input=check)[0]

        # Check if we've exhausted the split sequence or reached the end
        if depth >= len(self.split_sequence):
            return super().predict(instance.reshape(1, -1))[0]

        split_index = self.split_sequence[depth]

        # Recursively move left or right based on the threshold at the current node
        if instance[split_index] <= self.threshold:
            return self.left_tree._predict_instance(instance, depth=depth + 1)
        else:
            return self.right_tree._predict_instance(instance, depth=depth + 1)

    def visualize(self, feature_names=None, class_names=None) -> Source:
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
        dot_str, node = self.build_dot(feature_names, class_names, node=0, color_map=colorMap)
        dot += dot_str
        dot += '\n}'

        return graphviz.Source(dot, filename=None)

    def build_dot(self, feature_names, class_names, node=0, color_map=None):
        dot_str = ""
        try:
            feature_name = feature_names[self.split_index]
        except:
            feature_name = f"Feature {self.split_index}"
        threshold = self.threshold
        impurity = self.impurity  # Impurity at this node
        samples = self.n_node_samples  # Number of samples at this node

        class_index = np.argmax(self.value)  # Get the majority class index
        try:
            class_name = class_names[class_index]
        except:
            class_name = f"Class {class_index}"
        # Label for the internal node
        label = f'<{feature_name} ({self.split_index}) &le; {threshold:.2f}<br/>gini = {impurity:.3f}<br/>samples = {samples}>'

        my_fill = f'"{color_map[class_name]}"' if color_map else '"#fdfcff"'
        dot_str += f'{node} [label={label}, fillcolor={my_fill}] ;\n'
        # new_features = [feature_names[i] for i in range(len(feature_names)) if i != self.split_index]
        new_features = feature_names
        new_node = 0
        if getattr(self, 'left_tree', None):
            # Recursively build left and right trees
            left_dot_str, tree_size = self.left_tree.build_dot(new_features, class_names, node=node + 1,
                                                               color_map=color_map)
            # Add the left and right children to the parent node
            dot_str += f'{node} -> {node + 1};\n'
            # Add the left and right subtrees to the DOT string
            dot_str += left_dot_str
            # Update the node index for the next call
            new_node = node + tree_size
        if getattr(self, 'right_tree', None):
            right_child = new_node
            right_dot_str, tree_size = self.right_tree.build_dot(new_features, class_names, node=right_child,
                                                                 color_map=color_map)
            dot_str += right_dot_str
            dot_str += f'{node} -> {right_child};\n'

        # If this is a leaf node, generate a label with class distribution for the leaf
        if getattr(self, 'leaf_model', None):
            # Visualize the entire leaf model
            new_dot, tree_size = self.visualize_leaf_tree(node, feature_names, class_names, color_map)
            # dot_str += f'{node} -> {right_child};\n'
            node += tree_size
            dot_str += new_dot
        return dot_str, node

    def visualize_leaf_tree(self, node, feature_names, class_names, class_colors):
        # Generate the visualization of the leaf tree
        leaf_dot = export_graphviz(
            self.leaf_model,
            out_file=None,  # Return as a string
            feature_names=feature_names,
            class_names=class_names,
            rounded=True,
            special_characters=True  # Handle special characters in feature names
        )

        if class_colors:
            # Step 3: Use regex to apply colors to each node based on class
            for class_name, color in class_colors.items():
                leaf_dot = re.sub(
                    f'class *= *{class_name}>',
                    f'class = "{class_name}"> fillcolor="{color}"',
                    leaf_dot
                )

        def shift_index(match):
            index = int(match.group(1))
            return f" {index + node}"

        # Filter out lines that do not start with a node or edge definition
        filtered_lines = []
        for line in leaf_dot.splitlines():
            # Keep only lines that define a node or edge (node index or node -> connection)
            if '->' in line or re.match(r'\d+ +(?:\[.*\])*', line):
                # print(line)
                shifted_line = re.sub(r'(\b\d+\b)(?=\s*[\[;]| ->)', shift_index, line).strip()
                # print(shifted_line)
                # print('\n')
                filtered_lines.append(shifted_line)
        # Join the filtered lines back together into a DOT string
        filtered_dot = "\n".join(filtered_lines)

        return filtered_dot, self.leaf_model.tree_.node_count
