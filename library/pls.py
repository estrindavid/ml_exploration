# Partial Least Squares (PLS) regression implementation.
class PartialLeastSquares:

    # Calculate the mean of a list.
    def mean(self, arr):
        return sum(arr) / len(arr)
    
    # Center the matrix values (Input values), by subtracting the feature mean by the features current value.
    def center_matrix(self, X):
        # Separate the matrix into its features and the data/sample.
        self.n_features = len(X[0])
        self.n_data = len(X)

        # Calculate the mean of each feature (column).
        feature_means = [0] * self.n_features
        for j in range(self.n_features):
            for i in range(self.n_data):
                feature_means[j] += X[i][j]
            feature_means[j] /= self.n_data

        # Subtract the mean from each element in the matrix.
        centered_X = []
        for i in range(self.n_data):
            row = []
            for j in range(self.n_features):
                row.append(X[i][j]-feature_means[j])
            centered_X.append(row)

        return centered_X
    
    # Center the vector values (Output values).
    def center_vector(self, Y):
        #Calculate the mean of Y
        Y_mean = sum(Y)/len(Y)

        #Find the centered mean (Y - Y_mean) value for each output
        centered_Y = []
        for i in range(len(Y)):
            centered_Y.append(Y[i]-Y_mean)

        return centered_Y


    #Calculate the appropriate weight for each feature
    def compute_weights(self, centered_X, centered_Y):
        #Calculate the weight for each feature by multiplying the centered inputs (centered_X) for each feature with the centered desired output (centered_Y)
        weights = []
        for j in range(self.n_features):
            j_weight = 0
            for i in range(self.n_data):
                j_weight += centered_X[i][j] * centered_Y[i]
            weights.append(j_weight)
        
        return weights
        
    # Normalize the weights (scaling them down to a reasonable amount adding up to 1).
    def normalize(self, weights):
        # Use Euclidean norm.
        length = 0

        # Find the sum of the squares of the weights.
        for w in weights:
            length += w**2
        
        # If length is zero, return weights as they are.
        if length == 0:
            return weights

        # Square root the sums of the weights.
        length = length ** 0.5

        # Adjust the weights.
        normalized_weights = []
        for w in weights:
            normalized_weights.append(w/length)
        
        return normalized_weights

    # Create a component merging the features in centered_X into a singular component using weights.
    def create_component(self, centered_X, weights):
        # For each data point, compute the weighted sum of feature values.
        component = []
        for i in range(self.n_data):
            feature_value = 0
            for j in range(self.n_features):
                feature_value += centered_X[i][j] * weights[j]
            component.append(feature_value)

        # Return as 2D list (column vector shape).
        return [[value] for value in component]

    # Implement Partial Least Squares to create a single component out of multiple features.
    def transform(self, X, Y):
        # Center all data.
        X_centered = self.center_matrix(X)
        Y_centered = self.center_vector(Y)

        # Compute raw weights.
        raw_weights = self.compute_weights(X_centered, Y_centered)

        # Normalize the weights.
        normal_weights = self.normalize(raw_weights)

        # Build PLS component.
        component = self.create_component(X_centered, normal_weights)

        return component