# Arch-Technologies_Email-Spam-Classification
Month 1, Task 1

# Email Spam Classification Using Machine Learning

A machine learning project that classifies text messages as **spam** or **ham (not spam)**. The notebook covers the complete workflow, including data loading, cleaning, exploratory data analysis, text preprocessing, feature extraction, model training, evaluation, prediction, and model saving.

## Project Overview

Spam messages often contain promotional, fraudulent, or misleading content. This project builds and compares multiple supervised machine learning models to automatically identify spam messages.

The project was completed as part of a **Machine Learning Internship – Month 1 Task**.

## Dataset

The notebook uses the file:

```text
Email_dataset.csv
```

The original dataset contains:

- **5,572 rows**
- **5 columns**
- Two useful columns:
  - `v1`: Message label (`ham` or `spam`)
  - `v2`: Message text

After removing unnecessary columns and duplicate messages, the final dataset contains **5,169 unique messages**.

### Class Distribution

| Class | Messages |
|---|---:|
| Ham | 4,516 |
| Spam | 653 |

The dataset is imbalanced, so class balancing is applied to Logistic Regression and Linear SVM.

## Text Preprocessing

The preprocessing function performs the following operations:

- Converts text to lowercase
- Removes URLs
- Removes punctuation and special characters
- Removes extra spaces

TF-IDF is then used to convert the cleaned text into numerical features.

### TF-IDF Configuration

```python
TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)
```

This configuration uses both unigrams and bigrams and limits the vocabulary to the 5,000 most useful features.

## Models Used

Three machine learning models were trained and evaluated:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine

## Train-Test Split

The cleaned dataset was divided using an 80/20 stratified split.

| Dataset | Samples |
|---|---:|
| Training set | 4,135 |
| Testing set | 1,034 |

A fixed `random_state=42` was used for reproducibility.

## Model Performance

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 97.78% | 97.76% | 97.78% | 97.76% |
| Linear SVM | 97.78% | 97.74% | 97.78% | 97.73% |
| Multinomial Naive Bayes | 96.81% | 96.88% | 96.81% | 96.62% |

## Best Model

**Logistic Regression** was selected as the best-performing model because it achieved the highest weighted F1-score.

### Logistic Regression Classification Results

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Ham | 0.99 | 0.99 | 0.99 | 903 |
| Spam | 0.92 | 0.90 | 0.91 | 131 |

### Confusion Matrix

| Actual / Predicted | Ham | Spam |
|---|---:|---:|
| Ham | 893 | 10 |
| Spam | 13 | 118 |

The model correctly classified:

- **893 ham messages**
- **118 spam messages**

It misclassified:

- **10 ham messages as spam**
- **13 spam messages as ham**

## Sample Predictions

| Message | Prediction |
|---|---|
| Congratulations! You have won a free prize. Call now to claim. | Spam |
| Hi, can we meet tomorrow for the project discussion? | Ham |
| URGENT! Your account has been selected for a cash reward. | Spam |

---
## Author

**Affan Abdullah**
Computer Science
Machine Learning and Artificial Intelligence Enthusiast
**Arch Technologies Machine Learning Internship**.
