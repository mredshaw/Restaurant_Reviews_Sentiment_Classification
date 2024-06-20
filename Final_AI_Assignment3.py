import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

df = pd.read_csv('/Users/mikeredshaw/Documents/Schulich MBAN/AI in Business/Assignment 3 - Ratings/reviews.csv', sep='\t')

#Map the review score to the sentiment values
df['Sentiment'] = df['RatingValue'].apply(lambda x: 0 if x in [1, 2] else 1 if x == 3 else 2)

# Balance the dataset by dropping positive sentiment reviews, and keep only the relevant columns.
df = df.drop(df[df['Sentiment'] == 2].sample(n=1200, random_state=42).index)[['Sentiment', 'Review']]
df = df.reset_index(drop=True) 
# Split the dataset into training and validation sets (75% training, 25% validation)
train_df, valid_df = train_test_split(df, test_size=0.25, random_state=42)

train_df.to_csv('/Users/mikeredshaw/Documents/Schulich MBAN/AI in Business/Assignment 3 - Ratings/train.csv', index=False)
valid_df.to_csv('/Users/mikeredshaw/Documents/Schulich MBAN/AI in Business/Assignment 3 - Ratings/valid.csv', index=False)
train_df = pd.read_csv('/Users/mikeredshaw/Documents/Schulich MBAN/AI in Business/Assignment 3 - Ratings/train.csv')
valid_df = pd.read_csv('/Users/mikeredshaw/Documents/Schulich MBAN/AI in Business/Assignment 3 - Ratings/valid.csv')

# Define the SGD Classifier pipeline
sgd_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(random_state=42)),
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'tfidf__use_idf': (True, False),
    'clf__penalty': ['l2', 'l1', 'elasticnet'],
    'clf__alpha': [1e-2, 1e-3, 1e-4],
}

# Perform grid search on the training data
grid_search = GridSearchCV(sgd_pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(train_df['Review'], train_df['Sentiment'])

# Print the best parameters found by GridSearchCV
print("Best parameters found for grid-searched SGD:")
print(grid_search.best_params_)

# Use the best estimator to make predictions on the validation set
best_model = grid_search.best_estimator_
predictions = best_model.predict(valid_df['Review'])

# Performing cross-validation on the training data
cv_scores = cross_val_score(best_model, train_df['Review'], train_df['Sentiment'], cv=5, scoring='accuracy')

# Calculate and print metrics for the validation set
validation_accuracy = accuracy_score(valid_df['Sentiment'], predictions)
validation_f1 = f1_score(valid_df['Sentiment'], predictions, average='weighted')
conf_matrix = confusion_matrix(valid_df['Sentiment'], predictions)

print("\n",'='*100)
print('\nGrid-searched SGD Classifier Results:\n')
print(f'Accuracy: {validation_accuracy}\n')
print(f'F1 Score: {validation_f1}\n')
print(f'Cross-Validated Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})\n')
print('Confusion Matrix:')
print('          negative  neutral  positive')
print(f'negative   {conf_matrix[0,0]}        {conf_matrix[0,1]}        {conf_matrix[0,2]}')
print(f'neutral    {conf_matrix[1,0]}        {conf_matrix[1,1]}        {conf_matrix[1,2]}')
print(f'positive   {conf_matrix[2,0]}        {conf_matrix[2,1]}        {conf_matrix[2,2]}')