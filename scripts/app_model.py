# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import re
import string
import joblib

# Load in two datasets containing fake and true news
df_fake = pd.read_csv("./data/Fake.csv")
df_true = pd.read_csv("./data/True.csv")

# Assign binary class labels to fake news and true news
df_fake["class"] = 0
df_true["class"] = 1

# Split the two datasets into training data and manual testing data
df_fake_testing_data = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
df_true_testing_data = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)

# Label the manual testing data
df_fake_testing_data["class"] = 0
df_true_testing_data["class"] = 1

# Concatenate the two manual testing data sets into one
df_manual_testing = pd.concat([df_fake_testing_data, df_true_testing_data], axis = 0)

# Save the manual testing data set to csv file
df_manual_testing.to_csv("manual_testing_data.csv")

# Concatenate the two training data sets into one
df_marge = pd.concat([df_fake, df_true], axis =0 )

# Drop the title, subject, and date columns
df = df_marge.drop(["title", "subject","date"], axis = 1)

# Check for missing values
df.isnull().sum()

# Shuffle the rows of the dataframe
df = df.sample(frac = 1)

# Reset the index
df.reset_index(inplace = True)

# Drop the index column
df.drop(["index"], axis = 1, inplace = True)

# Define a function to preprocess the text data
def wordopt(text):
    # Lowercase the text
    text = text.lower()
    # Remove any text in square brackets
    text = re.sub('\[.*?\]', '', text)
    # Remove any non-word characters
    text = re.sub("\\W"," ",text) 
    # Remove any URLs
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # Remove any HTML tags
    text = re.sub('<.*?>+', '', text)
    # Remove any punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # Remove any line breaks
    text = re.sub('\n', '', text)
    # Remove any words containing numbers
    text = re.sub('\w*\d\w*', '', text)    
    return text


# Add the word optimization to the text column in the dataframe
df["text"] = df["text"].apply(wordopt)

# Store the text and class columns as x and y respectively
x = df["text"]
y = df["class"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Use TfidfVectorizer to vectorize the training and testing data
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# 1. Logistic Regression Model

# Train the model on the vectorized training data
LR = LogisticRegression()
LR.fit(xv_train,y_train)

# Predict class for the vectorized test data
pred_lr=LR.predict(xv_test)
# Calculate accuracy score for the test data
LR.score(xv_test, y_test)
# Print the classification report for the test data
print(classification_report(y_test, pred_lr))

# Save the model
joblib.dump(LR, 'LoR_model.pkl')
# Save the model
joblib.dump(vectorization, 'vectorization.pkl')
