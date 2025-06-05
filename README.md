# AI/ML Internship - Task 6: K-Nearest Neighbors (KNN) Classification - Iris Species Prediction

## Objective
The objective of this task was to understand and implement the K-Nearest Neighbors (KNN) algorithm for classification problems, including feature normalization, K selection, and decision boundary visualization.

## Dataset
The dataset used for this task is the [Iris.csv](Iris.csv) dataset, a classic dataset in machine learning used for classification tasks to distinguish between three species of Iris flowers.

## Tools and Libraries Used
* **Python**
* **Pandas:** For data loading and manipulation.
* **Scikit-learn:** For machine learning model implementation (KNeighborsClassifier, train-test split, MinMaxScaler, LabelEncoder) and evaluation metrics (accuracy_score, confusion_matrix, classification_report).
* **Matplotlib:** For creating static visualizations (accuracy vs. K plot, confusion matrix heatmap, decision boundary plot).
* **NumPy:** For numerical operations, especially for creating meshgrids for decision boundary visualization.
* **Seaborn:** For enhanced statistical graphics.

## KNN Classification Steps Performed:

### 1. Choose a Classification Dataset and Normalize Features
* Loaded the `Iris.csv` dataset.
* Separated features (sepal length, sepal width, petal length, petal width) from the target variable ('Species'). The 'Id' column was dropped.
* Encoded the categorical 'Species' target variable into numerical labels (0, 1, 2).
* **Normalized** all numerical features using `MinMaxScaler`. This scaling is crucial for KNN as it relies on distance calculations.
* Split the data into training (70%) and testing (30%) sets, ensuring stratification to maintain class proportions.
* **Outcome:** The dataset was successfully prepared and scaled, making it suitable for KNN classification.

### 2. Use KNeighborsClassifier from sklearn
* Initialized a `KNeighborsClassifier` with an initial `n_neighbors` value of 5.
* Trained the model on the scaled training data.
* Made predictions on the scaled test set.
* Calculated and displayed the initial test accuracy.
* **Outcome:** A baseline KNN model was established, providing a first look at its predictive performance.

### 3. Experiment with Different Values of K
* Iterated through a range of odd `K` values (1 to 29) to determine the optimal number of neighbors.
* For each `K`, a KNN model was trained, and its accuracy on the test set was recorded.
* A plot was generated showing the test accuracy versus different `K` values.
* **Outcome:** Identified the optimal `K` value (which was `K=1` in our run) that yielded the highest accuracy, demonstrating the importance of hyperparameter tuning in KNN.

### 4. Evaluate Model using Accuracy, Confusion Matrix
* Trained the KNN model again using the identified optimal `K` value.
* Evaluated its performance using:
    * **Accuracy Score:** Overall percentage of correct predictions.
    * **Confusion Matrix:** A table showing true positives, true negatives, false positives, and false negatives for each class.
    * **Classification Report:** Detailed metrics including precision, recall, and F1-score for each class.
* A heatmap visualization of the confusion matrix was generated.
* **Outcome:** Gained a comprehensive understanding of the model's performance, identifying specific strengths and weaknesses in classifying different Iris species.

### 5. Visualize Decision Boundaries
* To visualize the decision process, a KNN model was trained using only the two most discriminative features ('PetalLengthCm' and 'PetalWidthCm') and the optimal `K`.
* A meshgrid was created across the 2D feature space.
* The model predicted the class for each point on the meshgrid, defining the decision regions.
* A plot was generated showing these decision boundaries, with actual training and test data points overlaid.
* **Outcome:** Provided a clear visual understanding of how KNN partitions the feature space to classify new data points, highlighting the local nature of the `K=1` decision rule.

## Visualizations
The repository includes the following generated plots:
* `knn_accuracy_vs_k.png`: Plot showing KNN test accuracy versus different `K` values.
* `knn_confusion_matrix.png`: Heatmap visualization of the model's confusion matrix.
* `knn_decision_boundaries.png`: Plot illustrating the decision boundaries of the KNN model on two features.

## Conclusion
This task provided practical experience with the K-Nearest Neighbors algorithm, covering essential steps from data preprocessing and normalization to hyperparameter tuning and comprehensive model evaluation, including visual insights into decision boundaries.
