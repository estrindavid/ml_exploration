# Importing the required dependencies.
import random
from pls import PartialLeastSquares

# CrossValidator class for performing k-fold cross-validation on a given model.
# The class splits the dataset into k folds and trains/test the model k times to
# evaluate its accuracy more reliably.
class CrossValidator:
    
    def __init__(self,model,k=10, use_pls=False):
        self.model = model
        self.k = k
        self.use_pls = use_pls
    
    # Method takes an array and finds the mean of the values within it.
    def __mean(self, arr):
        sum = 0
        for term in arr:
            sum+=term

        return sum/len(arr)

    def validation(self, X, Y):
        # Use floor division to calculate the approximate size of each fold in the input data.
        fold_size = len(X) // self.k

        # Create and randomize a list full of indices representing the input columns.
        indices = list(range(len(X)))
        random.shuffle(indices)

        accuracies = []

        # Loop through the number of sections (k) that the models data is separated in, training and testing the accuracy.
        for i in range(self.k):

            # Calculate the start and end indices for the current fold.
            start = i * fold_size
            if i==self.k-1:
                end = len(X)
            else:
                end = start + fold_size

            # Assign indices to test and training data for this current fold.
            test_idx = indices[start:end]
            train_idx = indices[:start] + indices[end:]

            # Create a test and training set for the current training indices.
            X_train = [X[index] for index in train_idx]
            X_test = [X[index] for index in test_idx]
            Y_train = [Y[index] for index in train_idx]
            Y_test = [Y[index] for index in test_idx]

            # Apply PLS if necessary.
            if self.use_pls:
                pls = PartialLeastSquares()
                # Train PLS on the training data.
                X_train = pls.transform(X_train, Y_train)
                # Apply the same transformation to the test data.
                X_test = pls.transform(X_test, Y_test)


            # Train the model with the training set.
            self.model.fit(X_train, Y_train)

            # Test the model with the testing set.
            predictions = self.model.predict(X_test)
            
            # Find the mean of the results.
            result = []
            for index in range(len(predictions)):
                result.append(predictions[index]==Y_test[index])
            accuracy = self.__mean(result)

            # Add the results for this fold to the overall results for the model.
            accuracies.append(accuracy)

        return self.__mean(accuracies)