import numpy as np

class DecisionTree:

    def __init__(self, criterion="entropy", max_depth=None):
        '''
        Initializes a new decision tree with a given impurity measure and max depth

        Args:
            criterion (String): Which impurity measure to use. Default value is "entropy"
            max_depth (int): Maximum depth of the tree. Default value is "None"
        '''

        self.criterion = criterion
        self.max_depth = max_depth


    def fit(self, X, y):
        '''
        Fits a decision tree to a given dataset

        Args:
            X (np.array): Input features from dataset
            y (np.array): Target label from dataset
        '''

        feature_indices = list(range(X.shape[1]))
        self.tree = self._build_tree(X, y, feature_indices)


    def _build_tree(self, X, y, feature_indices, depth = 0):

        # Return leaf node with that label if all datapoints have same label
        if len(np.unique(y)) == 1:
            return y[0]

        # Return leaf node with most common label if all datapoints have identical feature values
        if np.all(X == X[0, :]):
            return self._find_majority(y)

        # Return leaf node with most common label when recursion reaches max depth
        if self.max_depth is not None and depth >= self.max_depth:
            return self._find_majority(y)

        # Find max info gain and best feature index and threshold to split at
        (max_info_gain, best_feature_idx, best_threshold
         ) = self._find_max_info_gain(X, y, feature_indices)

        # Return leaf node with most common label when no best feature index 
        # is found or split does not give additional information gain
        if best_feature_idx is None or max_info_gain <= 0:
            return self._find_majority(y)


        # Split tree at best feature index, found by checking best feature index against threshold
        left_split = X[:, best_feature_idx] <= best_threshold
        right_split = X[:, best_feature_idx] > best_threshold

        (X_left, y_left, X_right, y_right
         ) = (X[left_split], y[left_split], X[right_split], y[right_split])

        # Run function recursively for all branches
        return {"feature_idx": best_feature_idx,
                "threshold": best_threshold,
                "left": self._build_tree(X_left, y_left, feature_indices, depth + 1),
                "right": self._build_tree(X_right, y_right, feature_indices, depth + 1)}


    def _find_majority(self, y):

        # Find most common label by checking value counts
        values, counts = np.unique(y, return_counts=True)

        return values[np.argmax(counts)]


    def _find_max_info_gain(self, X, y, feature_indices):

        max_info_gain = -np.inf
        best_feature_idx = None
        best_threshold = None

        # Check all features, calculate values and thresholds and find info gain
        for feature_idx in feature_indices:
            values = np.unique(X[:, feature_idx])
            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                info_gain = self._find_info_gain(X, y, feature_idx, threshold)

                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return (max_info_gain, best_feature_idx, best_threshold)


    def _find_info_gain(self, X, y, feature_idx, threshold):

        # Find left and right splits for feature
        left_split = X[:, feature_idx] <= threshold
        right_split = X[:, feature_idx] > threshold

        # Find target label for each split
        y_left = y[left_split]
        y_right = y[right_split]

        # Return 0 if length of either label is 0
        if len(y_left) == 0 or len(y_right) == 0:
            return 0

        # Calculate impurity for parent node
        parent_impurity = self._calculate_impurity(y)

        # Calculate weighted impurity of node
        left_weight = len(y_left) / len(y)
        right_weight = len(y_right) / len(y)

        weighted_impurity = (left_weight * self._calculate_impurity(y_left) + 
                            right_weight * self._calculate_impurity(y_right))

        return parent_impurity - weighted_impurity


    def _calculate_impurity(self, y):

        # Find value counts for all target labels to calculate probabilities
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        # Use impurity criterion from initializer to calculate impurity
        if (self.criterion == "entropy"):
            return -np.sum(probabilities * np.log2(probabilities))
        elif (self.criterion == "gini"):
            return 1.0 - np.sum(probabilities**2)
        else:
            raise ValueError("Impurity criterion must be entropy or gini")

    

    def predict(self, X):
        '''
        Predicts target label based on input features

        Args:
            X (np.array): Input features
        
        Returns:
            datatype:
            - y (np.array): Target label
        '''

        predictions = []

        # Check all values in input features
        for x in X:
            predictions.append(
                self._predict_node(x, self.tree)
            )

        return np.array(predictions)

    def _predict_node(self, x, node):

        # Return leaf node immediately
        if not isinstance(node, dict):
            return node

        # Find feature index and threshold of node
        feature_idx = node["feature_idx"]
        threshold = node["threshold"]

        # Run function recursively until leaf node is found
        if x[feature_idx] <= threshold:
            return self._predict_node(x, node["left"])
        else:
            return self._predict_node(x, node["right"])