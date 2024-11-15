import kagglehub
import numpy as np
import pandas as pd
import os

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from RecursiveCustomDecisionTreeClassifier import RecursiveCustomDecisionTreeClassifier

# Download latest version
path = kagglehub.dataset_download("uciml/red-wine-quality-cortez-et-al-2009")

# print("Path to dataset files:", path)

files = os.listdir(path)
# print("Files in the dataset directory:", files)

# Load the CSV file (adjust the filename if different)
csv_file = [f for f in files if f.endswith('.csv')][0]  # Select the first CSV file
df = pd.read_csv(os.path.join(path, csv_file))

conditions = [
    (df['quality'] <= 4),
    (df['quality'] > 4) & (df['quality'] <= 6),
    (df['quality'] > 6)
]
choices = ['low', 'medium', 'high']

# Create a new column with the class labels
df['result'] = np.select(conditions, choices)

print(df['result'].value_counts())

df=df.drop("quality", axis=1)

# Display basic info to understand the dataset
print(df.columns)
# print(df.sample())

from sklearn.model_selection import train_test_split

# Separate features and target variable
X = df.drop(columns='result', axis=1)
y = df['result']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

seed = 42

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=seed)

seq=[0]
my_classifier = RecursiveCustomDecisionTreeClassifier(split_sequence=seq, max_depth=4)  # Initialize your custom classifier here
my_classifier.fit(X_train, y_train)

y_pred = my_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy * 100:.2f}%')

graph = my_classifier.visualize(X.columns, np.unique(y))

graph.render(
    f"wine_s{seed}_[{'_'.join(map(str, seq))}]-{round(accuracy * 100, 1)}",
    format="png", cleanup=True)




