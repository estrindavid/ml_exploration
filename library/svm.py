# SVM_Classifier class for building and training a Support Vector Machine (SVM) model.
# The class is used to classify data into two categories (binary classification) by
# finding the optimal hyperplane that separates the data points. 
class SVMClassifier():

    # Learning rate is the amount of change that we want to implement in the bias.
    # Number of iterations is the amount of times we want the model to iterate/go through the data.
    # The lambda is the regularization parameter.
    def __init__(self, learning_rate, num_of_iterations, lambda_parameter):
        self.learning_rate = learning_rate
        self.num_of_iterations = num_of_iterations
        self.lambda_parameter = lambda_parameter
    
    # Creating an array full of 0's of size n.
    def __zeros(self, x):
        return list(0 for _ in range(x))

    # Function that fits the dataset to SVM Classifier.
    def fit(self, X, Y):
        # m is the number of data points in the dataset (number of rows).
        # n is the number of input features (number of columns).
        # Calculate the number of rows and columns.
        self.m = len(X)
        self.n = len(X[0]) if self.m > 0 else 0

        # Initiating the weight value and bias value.
        self.w = self.__zeros(self.n)

        self.b = 0

        self.X = X
        
        self.Y = Y

        # Implementing the gradient descent algorithm for optimization.
        for i in range(self.num_of_iterations):
            self.update_weights()

    # Method that calculates the dot product of 2 vectors.
    def __dot_product(self, x, y):
        # Iterate over each element in the vectors and accumulate the product.
        dot_product = 0
        for index in range(len(x)):
            dot_product += x[index] * y[index]

        return dot_product
    
    # Method that updates the sign of each element in the input array.
    def __update_sign(self, arr):
        # Iterate through each element in the array and update its sign.
        new_list = []
        for item in arr:
            if item >= 0:
                # Positive values are represented as 1.
                new_list.append(1)
            else:
                # Negative values are represented as 0.
                new_list.append(0)

        return new_list

    # Used for updating the weight and bias in the model using stochastic gradient descent (SGD).
    def update_weights(self):
        # Label encoding: convert the labels to -1 or 1 for SVM classification.
        y_label=[]
        for item in self.Y:
            if (item <= 0):
                # Convert negative labels to -1.
                y_label.append(-1)
            else:
                # Convert the positive labels to 1.
                y_label.append(1)

        #Gradients (dw, db) using stochastic gradient descent (SGD).
        for index, x_i in enumerate(self.X):
            # Condition to check if the current data point is correctly classified.
            condition = y_label[index] * (self.__dot_product(x_i, self.w) - self.b) >= 1

            # Initialize dw (gradient for weights) and db (gradient for bias) to zero.
            dw = [0] * self.n
            db = 0

            # If the data point satisfies the margin condition, apply regularization.
            if condition:
                for i in range(len(self.w)):
                    dw[i] += 2 * self.lambda_parameter * self.w[i]
                db = 0

            # If the margin condition is violated, update weights based on the error.
            else:
                for i in range(len(self.w)):
                    dw[i] += (2 * self.lambda_parameter * self.w[i]) - (x_i[i] * y_label[index])
                db = y_label[index]

            # Update weights and bias using the gradients and learning rate.
            self.w = [w_i - self.learning_rate * dw_i for w_i, dw_i in zip(self.w, dw)]
            self.b -= self.learning_rate * db

    # Used for predicting the label for a given input value using the trained SVM model.
    def predict(self, X):
        # Iterate over each input data point in X
        outputs = []
        for x_i in X:
            # Calculate the output of the hyperplane equation (y = wx + b).
            output = self.__dot_product(x_i, self.w) + self.b
            outputs.append(output)

        # Converting data to 1 or 0 (Binary classification).
        y_hat = self.__update_sign(outputs)
        
        return y_hat