# Restaurant Reviews Sentiment Classification

## Project Overview
This project involves building a sentiment classifier to classify restaurant reviews. The goal is to train and test a Bag of Words (BoW) text classifier on review texts using RatingValue as labels. The ratings are binned into negative, neutral, and positive sentiments.

## Data
- **Dataset:** `reviews.csv`
- **Description:** Contains approximately 2000 restaurant reviews. The values in `reviews.csv` are tab-separated.

## Task
1. **Data Preprocessing:**
   - Load the `reviews.csv` file.
   - Bin the ratings into negative (0), neutral (1), and positive (2) sentiments.
   - Balance the dataset by dropping some positive ratings.
   - Split the data into training and validation sets.
   - Save the training and validation sets as `train.csv` and `valid.csv`.

2. **Model Training:**
   - Load `train.csv` and train a BoW text classifier.

3. **Model Evaluation:**
   - Load `valid.csv` and evaluate the model.
   - Print performance metrics including accuracy, F1-score, and confusion matrix.

## Deliverable
- A single Python file (`Sentiment_Classification.py`) that:
  1. Loads `reviews.csv`, preprocesses the data, splits it, and saves the files as `train.csv` and `valid.csv`.
  2. Loads `train.csv` and trains the model.
  3. Loads the validation data (`valid.csv`) and prints the performance metrics on the validation set.

## Usage
1. Ensure you have Python 3 and the necessary libraries installed.
2. Clone this repository: `git clone https://github.com/mredshaw/Restaurant_Reviews_Sentiment_Classification.git`
3. Run the Python script: `python Sentiment_Classification.py`

## Performance Metrics
The script will output:
- **Accuracy:** The accuracy of the model on the validation set.
- **F1-Score:** The F1-score of the model on the validation set.
- **Confusion Matrix:** The confusion matrix showing the classification results.

## Tools and Libraries Used
- Python
- Pandas
- Scikit-learn
- NLTK

## Key Insights
This project demonstrates the application of sentiment analysis on restaurant reviews using a balanced dataset and evaluates the model using multiple performance metrics to ensure robust classification.

