# ❤️ Heart Disease Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on predicting whether a patient is likely to have **heart disease** using Machine Learning algorithms.

The project follows a complete Machine Learning pipeline starting from data preprocessing and exploratory analysis to feature selection, outlier treatment, model training, hyperparameter tuning, and model evaluation using performance metrics such as **ROC Curve and AUC**.

---

## 📊 Dataset Information

The dataset contains:

* **304 rows**
* **14 columns**
* **13 independent features**
* **1 target variable**
* **No null values**
* **No missing data**

Since the dataset did not contain missing values, no missing-value imputation was required.

### Dataset Structure

The dataset consists of patient-related clinical attributes that are used to predict the presence or absence of heart disease.

The target variable represents the heart disease outcome.

---

## 🔄 Machine Learning Pipeline

The project follows the following workflow:

```text
Dataset
   ↓
Data Validation
   ↓
Train-Test Split
   ↓
Variable Transformation
(Yeo-Johnson)
   ↓
Outlier Detection & Removal
   ↓
Feature Selection
   ↓
Data Scaling
   ↓
Class Balancing
   ↓
Model Training
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
ROC Curve & AUC
   ↓
Best Model Selection
   ↓
Heart Disease Prediction
```

---

## 1️⃣ Data Validation

Initially, the dataset was checked for:

* Number of rows and columns
* Missing values
* Null values
* Data types
* Duplicate records
* Feature distributions

The dataset contains **304 records and 14 columns**, and there are **no null or missing values**.

Therefore, no missing-value treatment was required.

---

## 2️⃣ Train-Test Split

The dataset was divided into training and testing datasets using an **80:20 ratio**.

The independent variables were separated from the target variable.

### Training Data

Approximately:

* **243 rows**
* **13 input features**

### Testing Data

Approximately:

* **61 rows**
* **13 input features**

The target variable (`y`) contains one column.

```python
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

> **Note:** Since 304 × 20% = 60.8, the split normally results in approximately **61 test rows and 243 training rows**.

---

# 3️⃣ Categorical Data Handling

After examining the dataset, there were no categorical features requiring traditional categorical encoding in the preprocessing stage.

Therefore, encoding techniques such as:

* One-Hot Encoding
* Label Encoding

were not required at this stage.

---

# 4️⃣ Variable Transformation

After splitting the dataset, **Yeo-Johnson transformation** was applied to the numerical variables.

### Why Yeo-Johnson Transformation?

Yeo-Johnson transformation is useful for transforming numerical variables and reducing skewness.

It has an advantage over some other transformations because it can handle **zero and negative values**.

The transformation was applied to improve the distribution of the variables and help reduce the influence of extreme values.

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method="yeo-johnson")

X_train_transformed = pt.fit_transform(X_train)
X_test_transformed = pt.transform(X_test)
```

The transformer was **fitted only on the training data** and then used to transform the test data.

---

# 5️⃣ Outlier Detection and Removal

After variable transformation, outliers were identified and treated.

The purpose of outlier treatment was to reduce the influence of extreme observations on the Machine Learning models.

The outlier detection process was performed on the training data, and the identified extreme observations were removed according to the project's preprocessing strategy.

This step helps improve the quality of the data before feature selection and model training.

---

# 6️⃣ Feature Selection

Feature selection was performed to identify useful features and remove features that provide little or no information to the Machine Learning models.

Three feature-selection techniques were considered.

---

## 🔹 6.1 Constant Feature Removal

The first technique was the **constant feature method**.

A constant feature has the same value for every observation.

Therefore, its standard deviation is:

```text
Standard Deviation = 0
```

Such a feature does not provide any useful information to the model.

Therefore, features with:

```text
Standard Deviation = 0
```

were removed.

### Example

```text
Feature A
10
10
10
10
10
```

Since there is no variation:

```text
Std = 0
```

The feature can be removed.

---

## 🔹 6.2 Quasi-Constant Feature Removal

The second technique was **quasi-constant feature removal**.

A quasi-constant feature has very little variation in its values.

In this project, features having very low variation were investigated using a threshold of approximately **0.1**, according to the preprocessing strategy.

Such features contribute very little information and can potentially be removed.

> **Important:** In `VarianceThreshold`, the parameter `threshold=0.1` refers to **variance**, not standard deviation. If your implementation uses `VarianceThreshold(threshold=0.1)`, describe it as a **variance threshold of 0.1** rather than a standard-deviation threshold.

---

## 🔹 6.3 Correlation-Based Feature Selection

The third feature-selection technique was based on **correlation**.

Correlation was analyzed to identify relationships between features and the target variable as well as relationships among independent variables.

Highly correlated redundant features can provide overlapping information.

Therefore, correlation analysis was used to identify features that could potentially be removed while retaining important predictive information.

---

# 7️⃣ Feature Scaling

After feature transformation, outlier treatment, and feature selection, feature scaling was performed.

Scaling ensures that numerical features are brought to a comparable scale.

This is particularly important for algorithms that are sensitive to feature magnitude.

The scaler was fitted on the training data and then applied to the test data to prevent data leakage.

---

# 8️⃣ Handling Class Imbalance

The distribution of the target classes was analyzed before model training.

If the classes were imbalanced, a balancing technique was applied to the training data.

The balancing process was performed only on the training data so that the test dataset remained representative of unseen data.

---

# 9️⃣ Machine Learning Models

Multiple Machine Learning algorithms were trained and evaluated for heart disease prediction.

The models were compared using appropriate evaluation metrics rather than relying only on accuracy.

The project includes models such as:

* Logistic Regression
* K-Nearest Neighbors
* Support Vector Classifier
* Decision Tree
* Random Forest
* Gradient Boosting
* Other applicable classification algorithms

---

# 🔟 Hyperparameter Tuning

Hyperparameter tuning was performed to find suitable parameters for the Machine Learning models.

**GridSearchCV** was used to evaluate different combinations of hyperparameters using cross-validation.

Example:

```python
GridSearchCV(
    estimator=model,
    param_grid=parameter,
    scoring="accuracy",
    cv=5
)
```

The best parameters obtained through cross-validation were then used to train the final model.

---

# 1️⃣1️⃣ Model Evaluation

The trained models were evaluated using multiple performance metrics.

The evaluation included:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve
* AUC Score

Using multiple metrics provides a more complete understanding of model performance.

---

# 📈 ROC Curve and AUC

The **Receiver Operating Characteristic (ROC) curve** was used to evaluate the classification performance of the models.

The **Area Under the Curve (AUC)** represents the model's ability to distinguish between the two target classes.

The ROC curves of different models were compared to understand their classification performance.

```text
True Positive Rate
        ↑
        |
        |       ______
        |     /
        |   /
        | /
        |________________→
          False Positive Rate
```

---

# 🏆 Final Model

After training and evaluating the different Machine Learning algorithms, the models were compared using their evaluation metrics, particularly **ROC-AUC**, along with other classification metrics.

The model selected for deployment was the model that performed best according to the project's evaluation criteria.

> The final model should be stated here after your actual ROC-AUC and validation results are confirmed.

---

# 🌐 Deployment

The trained Machine Learning model can be integrated with a web application where users can enter patient-related clinical information.

The application processes the input using the same preprocessing pipeline used during training and then generates a heart disease prediction.
# 🌐 Live Web Application

The Heart Disease Prediction application is deployed using **Render**.

🔗 **https://heart-predection-1-qrfo.onrender.com**


```



### Application Flow

```text
User Input
    ↓
Data Preprocessing
    ↓
Yeo-Johnson Transformation
    ↓
Feature Selection
    ↓
Scaling
    ↓
Trained ML Model
    ↓
Prediction
    ↓
Heart Disease / No Heart Disease
```

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* SciPy
* Imbalanced-learn

### Machine Learning

* Classification Algorithms
* Feature Selection
* Yeo-Johnson Transformation
* Outlier Detection
* Feature Scaling
* Hyperparameter Tuning
* ROC-AUC Analysis

### Development Environment

* PyCharm
* Jupyter Notebook / Python
* Flask (for web deployment, if applicable)

---

# 📁 Project Structure

```text
Heart_Disease_Prediction/
│
├── main.py
├── heart.csv
│
├── logger_code.py
├── yeo_timing.py
├── feature_sele.py
│
├── model/
│   └── trained_model.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── README.md
```

---

# 🎯 Project Objectives

The main objectives of this project are:

1. To analyze clinical data related to heart disease.
2. To preprocess the dataset effectively.
3. To transform numerical variables using Yeo-Johnson transformation.
4. To detect and handle outliers.
5. To remove unnecessary or low-information features.
6. To train multiple Machine Learning classification algorithms.
7. To optimize models using hyperparameter tuning.
8. To compare models using ROC-AUC and other evaluation metrics.
9. To develop a Machine Learning model capable of predicting heart disease.

---

# 💡 Key Learning Outcomes

Through this project, I gained practical experience in:

* Data preprocessing
* Train-test splitting
* Yeo-Johnson transformation
* Outlier detection
* Feature selection
* Feature scaling
* Handling class imbalance
* Machine Learning classification
* Hyperparameter tuning
* Cross-validation
* ROC curve analysis
* AUC evaluation
* Machine Learning pipeline development
* Model deployment

---



### Project Domain

**Machine Learning**

### Project Title

**Heart Disease Prediction Using Machine Learning**

---

