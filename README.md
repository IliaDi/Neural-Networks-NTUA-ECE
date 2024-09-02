### HW1 SMALL

#### **Goal:**
The objective of this homework is to classify EEG data into two mental states: planning and relaxation. The dataset used contains EEG measurements taken from a healthy right-handed individual during two phases—one of relaxation and another where the individual was asked to imagine moving fingers without actual movement.

#### **Methods:**

1. **Data Preparation:**
   - **Dataset Description:** The dataset comprises 182 samples with 13 features each, representing different EEG frequency bands (Delta, Theta, Alpha, Beta, Gamma, and Mu).
   - **Data Conversion:** The original text file is converted into a CSV format, and unnecessary columns (like an extra one created during conversion) are removed.
   - **Exploratory Data Analysis:** The dataset is checked for missing values, data types, and class distribution. The class distribution is imbalanced, with 71.4% of samples in class 1 and 28.6% in class 2.

2. **Baseline Classification:**
   - **Baseline Models:** Various dummy classifiers (`uniform`, `constant`, `most_frequent`, `stratified`) are implemented to establish baseline performance. These models provide a point of comparison for more sophisticated classifiers.
   - **K-Nearest Neighbors (KNN):** A KNN classifier is used with default parameters to compare against the dummy classifiers. Metrics like accuracy, confusion matrix, and F1 scores (both micro and macro) are computed.

3. **Model Evaluation:**
   - **Cross-Validation:** A custom 10-fold cross-validation is implemented to optimize the `k` parameter in the KNN classifier. This step aims to find the optimal number of neighbors for the classifier.
   - **Preprocessing Techniques:**
     - **Without Preprocessing:** The KNN classifier is evaluated without any data preprocessing.
     - **Variance Threshold & Standard Scaling:** The data is first reduced using a Variance Threshold selector and then scaled using a Standard Scaler before applying the KNN classifier. This step aims to improve model performance by removing low-variance features and normalizing the data.

4. **Performance Metrics:**
   - **Confusion Matrix:** Used to visualize the classification results and identify where the model is making errors.
   - **F1 Scores (Micro and Macro):** Calculated for each classifier to provide insights into precision and recall. The difference between micro and macro F1 scores helps in understanding the impact of class imbalance on model performance.

5. **Model Optimization:**
   - **Comparison of Architectures:** Different preprocessing techniques are evaluated to determine which leads to the best KNN classifier performance.
   - **Hyperparameter Tuning:** The optimal value of `k` in the KNN classifier is determined through cross-validation.

---

### HW1 BIG

#### **Overview**
The homework focuses on classifying emails as spam or not spam using the Spambase dataset. The dataset consists of 4,601 email samples with 57 numerical features that include word frequencies, character frequencies, and capital letter statistics. The tasks involve evaluating different classifiers, optimizing their performance, and analyzing the results.

#### **Goals**
1. **Dataset Introduction and Preparation**
   - Load and preprocess the Spambase dataset.
   - Split the data into training and test sets.
   - Calculate and display class frequencies and percentages.

2. **Baseline Classification**
   - Implement and evaluate several baseline classifiers including Dummy Classifiers, Gaussian Naive Bayes, K-Nearest Neighbors (KNN), and Multi-Layer Perceptron (MLP).
   - Measure and compare their performance using accuracy, confusion matrix, and precision-recall metrics.

3. **Optimization of Classifiers**
   - Optimize hyperparameters for the classifiers using Grid Search.
   - Implement preprocessing steps such as feature selection, scaling, and dimensionality reduction.
   - Evaluate and compare the optimized models' performance.

4. **Execution Time Analysis**
   - Measure and compare the execution time for fitting and predicting with different classifiers.

#### **Methods**

1. **Dataset Introduction and Preparation**
   - **Loading Data:** Use `pandas` to load the dataset from a URL.
   - **Data Splitting:** Split the data into training (70%) and testing (30%) sets using `train_test_split`.
   - **Class Distribution:** Compute and display class frequencies and percentages.

2. **Baseline Classification**
   - **Dummy Classifiers:** Test strategies like uniform random predictions, constant predictions, and most frequent label predictions.
   - **Gaussian Naive Bayes:** Use the Gaussian Naive Bayes classifier to make predictions.
   - **KNN:** Apply the K-Nearest Neighbors algorithm with default parameters.
   - **MLP:** Evaluate the performance of a Multi-Layer Perceptron with a single hidden layer.
   - **Evaluation Metrics:** Compute accuracy, confusion matrix, and precision-recall scores (micro and macro averages).

3. **Optimization of Classifiers**
   - **Hyperparameter Tuning:** Use `GridSearchCV` to find the best hyperparameters for Dummy Classifiers, Gaussian Naive Bayes, KNN, and MLP.
   - **Preprocessing Pipelines:** Implement pipelines with preprocessing steps including feature selection (VarianceThreshold), scaling (StandardScaler), oversampling (RandomOverSampler), and dimensionality reduction (PCA).
   - **Pipeline Creation:** Define and fit pipelines for each classifier and evaluate their performance with optimized parameters.

4. **Execution Time Analysis**
   - **Time Measurement:** Measure the time taken for fitting and predicting with each model.
   - **Comparison:** Compare execution times and performance metrics before and after optimization.

The homework involves both hands-on machine learning tasks and performance optimization, providing practical experience with classification models and their evaluation.

---

### HW2

**Goal:** 

The main objective is to implement and optimize a movie recommendation system based on content using TF-IDF (Term Frequency-Inverse Document Frequency) and a Self-Organizing Map (SOM) for visualization and clustering.

**Steps:**

1. **Dataset Installation and Preparation:**
   - Install necessary Python libraries (`numpy`, `pandas`, `nltk`, `scikit-learn`, `joblib`, and `somoclu`).
   - Import and prepare the dataset, which consists of movie descriptions, titles, categories, and summaries.

2. **TF-IDF Transformation:**
   - Convert the movie summaries into TF-IDF vectors using `TfidfVectorizer`. This step transforms the textual data into numerical form suitable for similarity calculations.

3. **Recommendation System Implementation:**
   - Implement a function `content_recommender` that:
     - Computes cosine similarity between a target movie and all other movies based on their TF-IDF representations.
     - Sorts the movies based on similarity scores and returns the most similar movies.
     - Prints information about the target movie and its top recommendations.

4. **Optimization:**
   - Optimize the TF-IDF vectorizer parameters to improve recommendation quality. This involves experimenting with parameters like `max_df`, `min_df`, `stop_words`, `ngram_range`, and `max_features`.

5. **Persistence:**
   - Use `joblib` to save and load the TF-IDF matrix (`corpus_tf_idf`) to avoid recalculating it multiple times.

6. **Self-Organizing Map (SOM) for Clustering:**
   - Use the SOM library to visualize and cluster the movies based on TF-IDF vectors and binary category features.
   - Train a SOM model and apply K-Means clustering to group movies into clusters.
   - Visualize the SOM using the U-matrix and print information about clusters, such as the number of neurons in each cluster.

7. **Visualization and Reporting:**
   - Print and visualize the U-matrix and clusters.
   - Provide detailed reports on movie categories within clusters and the number of movies in each cluster.

#### Methods:

- **TF-IDF Vectorization:** Transforms movie summaries into a numerical representation that captures the importance of each term relative to the entire corpus.
- **Cosine Similarity:** Measures the similarity between movies based on their TF-IDF vectors.
- **Self-Organizing Maps (SOM):** A type of neural network used for unsupervised learning, which helps in visualizing and clustering high-dimensional data.
- **K-Means Clustering:** Applied to cluster the SOM map into distinct groups.

This approach involves both the content-based filtering for recommendations and unsupervised learning techniques for clustering and visualization.

---

### HW3

**Goal:**

The goal is to build and evaluate a convolutional neural network (CNN) model for image classification on the CIFAR-100 dataset, using both custom-built models and transfer learning with VGG16.

**Steps:**

1. **Dataset Preparation:**
   - **Data Splitting and Normalization:**
     - Split the CIFAR-100 dataset into training, validation, and test sets.
     - Normalize pixel values by scaling them to the range [0,1].
   - **Data Augmentation and Prefetching:**
     - Use TensorFlow’s `tf.data.Dataset` API to create training, validation, and test datasets with prefetching for performance optimization.

2. **Model Definition and Training:**
   - **Custom CNN Model:**
     - Define a simple CNN from scratch with several convolutional and pooling layers, followed by dense layers.
     - Compile and train this model using the prepared datasets.
   - **Transfer Learning with VGG16:**
     - Use the pre-trained VGG16 model (without the top classification layer) and add custom layers for CIFAR-100 classification.
     - Fine-tune the model by training only the newly added layers or the entire model depending on the setup.

3. **Evaluation and Visualization:**
   - **Training Diagnostics:**
     - Plot training and validation loss and accuracy curves to visualize the model’s learning progress.
   - **Model Evaluation:**
     - Evaluate the model on the test set and print metrics like loss and accuracy.

4. **Experimentation:**
   - **Try Different Architectures:**
     - Optionally, test and compare different model architectures, either by modifying the custom CNN or using different transfer learning models.

#### Methods:

- **Data Preparation:**
  - **Normalization:** Scaling image pixel values to [0,1] for better training performance.
  - **Prefetching:** Using TensorFlow’s `tf.data.Dataset` with prefetching to optimize data loading and training speed.

- **Model Building:**
  - **Custom CNN:** Creating a simple CNN with convolutional, pooling, and dense layers.
  - **Transfer Learning:** Using VGG16 pre-trained on ImageNet, with additional layers for CIFAR-100 classification, and optionally fine-tuning various layers.

- **Training and Evaluation:**
  - **Training:** Using TensorFlow’s model fitting functions to train the models with specified epochs and batch sizes.
  - **Evaluation:** Assessing the model’s performance on the test set and visualizing learning curves.

- **Visualization:**
  - **Diagnostic Plots:** Creating plots to visualize training and validation loss and accuracy over epochs.

This approach ensures a thorough evaluation of both custom-built and pre-trained models, with careful attention to data preparation, training processes, and model performance.
