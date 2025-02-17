# Text Classification Project

## Overview

This project implements a **text classification system** using multiple machine learning models, including **Random Forest, Multinomial Naïve Bayes, Linear SVC, and Logistic Regression**. The system is designed to classify text documents into predefined categories using **TF-IDF vectorization**, **feature selection techniques**, and **various classification models**.

It supports multiple classification methods, leveraging **scikit-learn, TensorFlow/Keras, and NLP techniques** such as **lemmatization and stopword removal**. The system is built with a modular structure, enabling flexible integration with different classifiers.

## Features

- **Text Preprocessing:** Tokenization, stopword removal, and TF-IDF vectorization.
- **Feature Selection:** Uses **Chi-Square** for selecting important features.
- **Multiple Classification Models:** Supports **Random Forest, Multinomial Naïve Bayes, Logistic Regression, and Linear SVC**.
- **Evaluation Metrics:** Generates **confusion matrix, classification report, and cross-validation scores**.
- **Support for Keras Models:** Includes a **deep learning classifier** using Keras.
- **Pipeline Implementation:** Automates vectorization, feature selection, and classification in a single step.

## Project Structure

```
├── Datasets/                          # Contains text classification datasets
├── Documents/                         # Documentation and related files
├── FeatureVectorClassifier/           # Feature-based classification models
│   ├── infoclassifier.py              # TF-IDF vectorizer and multiple classifiers
├── KerasClassifier/                   # Neural network-based classification
├── MultinomialNaiveBayesClassifier/   # Naïve Bayes classification
│   ├── NaiveBayesMulti.py             # Multinomial Naïve Bayes classifier
├── RandomForestClassifier/            # Random Forest-based classification
│   ├── randomforest.py                # Random Forest model implementation
├── preliminaries/                     # Preprocessing utilities
│   ├── classifier.py                   
│   ├── contractions.py                 
│   ├── normalizer.py                    
│   ├── preprocessor.py                 # Preprocessing pipeline
│   ├── scrapper.py                     
├── __pycache__/                        # Compiled Python cache files
├── main.py                             # Entry point for running classification
├── model.h5                            # Saved deep learning model
├── sentence.csv                        # Example dataset
├── .gitignore                          # Git ignore file
```

## Installation

1. **Clone the repository**
```bash
git clone <repo-url>
cd text-classification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the main script**
```bash
python main.py
```

## Data Preprocessing

- Text data is preprocessed using `preprocessor.py` which:
  - Removes stopwords
  - Lemmatizes words
  - Converts text to lowercase
  - Cleans special characters and digits

## Model Training & Evaluation

### **1. TF-IDF Vectorization**
TF-IDF is used to convert text data into numerical feature vectors:
```python
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=4, ngram_range=(1,2))
features = vectorizer.fit_transform(df['cleaned']).toarray()
```

### **2. Feature Selection (Chi-Square Test)**
Used to determine the most relevant features:
```python
from sklearn.feature_selection import chi2
N = 8
for tag, cat_id in sorted(cat_to_id.items()):
    features_chi2 = chi2(features, labels == cat_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(vectorizer.get_feature_names())[indices]
    print(f"Most Correlated Unigrams for {tag}: {feature_names[-N:]}")
```

### **3. Cross-Validation (5-Fold CV)**
To compare models and measure performance:
```python
from sklearn.model_selection import cross_val_score
models = [RandomForestClassifier(), LinearSVC(), MultinomialNB(), LogisticRegression()]
cv_scores = [cross_val_score(model, features, labels, scoring='accuracy', cv=5) for model in models]
```

### **4. Model Evaluation**
Confusion matrix visualization using Seaborn:
```python
import seaborn as sns
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt='d')
plt.show()
```

### **5. Predictions on Unseen Data**
```python
model = LinearSVC().fit(features, labels)
new_text = ["I recently moved to a new house in Switzerland."]
prediction = model.predict(vectorizer.transform(new_text))
print("Predicted Category:", prediction)
```

## Results

| Model                  | Mean Accuracy | Standard Deviation |
|------------------------|--------------|--------------------|
| RandomForest          | 84.5%        | 3.2%               |
| Linear SVC            | 88.7%        | 2.5%               |
| Multinomial Naïve Bayes | 82.3%        | 3.1%               |
| Logistic Regression   | 86.9%        | 2.8%               |

## Future Improvements

- **Deep Learning Expansion:** Implement a **bi-LSTM or Transformer-based model** for text classification.
- **Hyperparameter Tuning:** Optimize classifier parameters using **GridSearchCV**.
- **Data Augmentation:** Increase dataset size for better generalization.
- **Deployment:** Wrap the classification model into an **API** for real-world usage.

## Conclusion

This project provides a robust implementation of **text classification** using **multiple ML models**. It supports **data preprocessing, feature engineering, and model evaluation**, ensuring high accuracy for document classification.

**Contributions & feedback are welcome!** 🚀

