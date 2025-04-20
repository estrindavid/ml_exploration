# Comparison between SVM classifier without PLS and SVM classifier with PLS.

# Import necessary dependencies.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'library')))
from pls import PartialLeastSquares
from svm import SVMClassifier
from cross_validation import CrossValidator


# Sample data taken from ChatGPT:
# biomarker_1 (Blood Pressure), biomarker_2 (Heart Rate), biomarker_3 (Sleep Hours), biomarker_4 (Stress Level), biomarker_5 (Age)
X = [
    [130, 80, 7, 3, 45], [125, 75, 6, 4, 50], [140, 85, 5, 8, 33], [120, 70, 8, 2, 60],
    [135, 82, 6, 7, 28], [110, 65, 9, 1, 22], [115, 72, 8, 4, 30], [145, 90, 5, 9, 55],
    [138, 84, 6, 5, 42], [125, 76, 7, 3, 38], [137, 86, 4, 6, 29], [132, 78, 5, 8, 40],
    [128, 79, 6, 3, 35], [120, 74, 7, 4, 60], [135, 83, 8, 9, 25], [140, 85, 6, 7, 33],
    [125, 77, 9, 2, 45], [130, 80, 5, 7, 60], [138, 82, 6, 5, 32], [115, 68, 8, 3, 55],
    [132, 84, 7, 4, 50], [125, 73, 7, 6, 47], [140, 88, 6, 9, 22], [133, 80, 6, 5, 50],
    [127, 75, 8, 6, 39], [118, 69, 9, 2, 31], [135, 84, 6, 7, 45], [121, 70, 8, 5, 53],
    [133, 81, 5, 9, 28], [140, 86, 6, 6, 44], [125, 75, 6, 4, 60], [127, 77, 7, 5, 35],
    [138, 90, 5, 8, 40], [140, 83, 8, 3, 50], [125, 80, 7, 4, 29], [138, 85, 6, 7, 47],
    [130, 76, 8, 9, 42], [118, 69, 7, 6, 55], [130, 79, 5, 4, 34], [138, 85, 8, 7, 39],
    [133, 81, 6, 3, 28], [127, 74, 6, 5, 50], [135, 82, 7, 9, 30], [125, 75, 9, 4, 43],
    [140, 86, 5, 8, 33], [122, 70, 8, 3, 54], [135, 80, 7, 7, 27], [130, 75, 8, 9, 40],
    [140, 84, 5, 6, 50], [138, 87, 6, 5, 35], [125, 78, 7, 3, 43], [137, 83, 6, 9, 44],
    [133, 79, 8, 4, 39], [128, 77, 7, 6, 30], [132, 82, 6, 3, 41], [140, 85, 5, 7, 33],
    [125, 74, 7, 8, 47], [138, 80, 8, 5, 28], [135, 84, 6, 3, 38], [120, 70, 8, 2, 50],
    [125, 75, 7, 6, 36], [140, 88, 5, 7, 52], [138, 81, 9, 5, 29], [130, 76, 6, 9, 60],
    [125, 78, 7, 3, 35], [138, 86, 5, 8, 40], [130, 74, 7, 2, 50], [137, 84, 6, 4, 32],
    [128, 75, 6, 9, 39], [132, 79, 8, 6, 44], [133, 81, 5, 7, 30], [125, 77, 7, 8, 25],
    [140, 89, 5, 4, 50], [120, 70, 6, 9, 45], [135, 82, 8, 5, 50], [130, 77, 6, 3, 43],
    [125, 79, 7, 6, 30], [140, 88, 5, 7, 55], [138, 83, 6, 5, 41], [130, 76, 8, 6, 28],
    [125, 75, 7, 4, 39], [133, 81, 6, 3, 40], [140, 87, 8, 5, 34], [128, 78, 7, 6, 33],
    [135, 80, 5, 9, 45], [120, 70, 8, 2, 28], [133, 84, 6, 4, 48], [140, 86, 5, 8, 50],
    [125, 75, 6, 7, 34], [137, 83, 8, 3, 36], [130, 79, 7, 6, 31], [138, 81, 6, 5, 37]
]

# 0 = Not depressed, 1 = depressed
Y = [
    0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 
    0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 
    1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 
    0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 
    0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 
    0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0
]



# Creates an instance of the SVM classifier.
svm = SVMClassifier(learning_rate=0.01, num_of_iterations=1000, lambda_parameter=0.1)

# # Perform cross-validation for SVM without using PLS.
cv_svm = CrossValidator(model=svm, k=10)
accuracy_svm = cv_svm.validation(X, Y)
print(f"Cross-validated Accuracy of SVM without PLS: {accuracy_svm * 100:.2f}%")

# Create a partial least squares instance and transform the features of the input into a single feature.
pls = PartialLeastSquares()

# Apply PLS transformation to X.
X_transformed = pls.transform(X, Y)

# Cross-validation for SVM after PLS transformation.
cv_svm_pls = CrossValidator(model=svm, k=10,use_pls=True)
accuracy_svm_pls = cv_svm_pls.validation(X_transformed, Y)
print(f"Cross-validated Accuracy of SVM with PLS: {accuracy_svm_pls * 100:.2f}%")