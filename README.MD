# Cyberbullying Classification Model

This project aims to build a machine learning model to classify cyberbullying content on social media based on tweets. The goal is to predict different types of cyberbullying, including age, ethnicity, gender, religion, other cyberbullying, and non-cyberbullying.

## Project Overview

- **Objective**: Create a classification model to predict cyberbullying types in tweets.
- **Dataset**: The dataset consists of tweets labeled with the type of cyberbullying they contain. The types include:
  - Age
  - Ethnicity
  - Gender
  - Religion
  - Not Cyberbullying
  - Other Cyberbullying

- **Algorithms Used**: 
  - Naïve Bayes
  - Logistic Regression
  - Decision Tree
  - Random Forest

- **Key Features**: 
  - Text preprocessing (including tokenization, stemming, and stopword removal)
  - TF-IDF vectorization for feature extraction
  - Evaluation using confusion matrix, classification report, and accuracy score

## Installation

To run this project, you need to have Python installed along with the required libraries. You can install the dependencies using the `requirements.txt` file.

### Steps to Install:

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/RohithgowdaM/cyberbullying-classification.git
    ```

2. Navigate to the project directory:
    ```bash
    cd cyberbullying-classification
    ```

3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Data Preprocessing**: The script processes the tweet data by performing tasks like tokenization, lemmatization, and stopword removal.
2. **Model Training and Evaluation**: Different machine learning models (Naïve Bayes, Logistic Regression, Decision Tree, Random Forest) are trained, and their performance is evaluated using accuracy, confusion matrix, and classification report.
3. **Visualization**: The confusion matrices for each model are displayed using heatmaps for better understanding of model performance.

To run the project, you can execute the main script (e.g., `program.py` or `main.py`).

```bash
python main.py
```