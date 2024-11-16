import kagglehub
import numpy as np
import pandas as pd
import os

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from AdaptiveDecisionTreeClassifier import AdaptiveDecisionTreeClassifier
dataset_path = "C:/Users/Mikhail23/.cache/kagglehub/datasets/uciml/red-wine-quality-cortez-et-al-2009/versions/2"
if not os.path.exists(dataset_path):
    dataset_path = kagglehub.dataset_download("uciml/red-wine-quality-cortez-et-al-2009")

files = os.listdir(dataset_path)
csv_file = [f for f in files if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(dataset_path, csv_file))

conditions = [
    (df['quality'] <= 5),
    (df['quality'] > 4) & (df['quality'] <= 6),
    (df['quality'] > 6)
]
choices = ['bad', 'ok', 'good']

# Create a new column with the class labels
df['result'] = np.select(conditions, choices)

df=df.drop("quality", axis=1)

from sklearn.model_selection import train_test_split

# Separate features and target variable
X = df.drop(columns='result', axis=1)
print(list(X.columns))
y = df['result']

print("y \n", y.value_counts(normalize=True), '\n')

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

seed = 42

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=seed)

seq=[0, 3, 2]
d = 6
my_classifier = AdaptiveDecisionTreeClassifier(
    split_feature_order=seq, feature_names=X.columns, max_depth=d)
my_classifier.fit(X_train, y_train)

y_pred = my_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy * 100:.2f}%')

decoded_predictions = pd.Series(encoder.inverse_transform(y_pred))
print("predictions \n", decoded_predictions.value_counts(normalize=True), '\n')

graph = my_classifier.visualize(class_names=np.unique(y))

graph.render(
    f"graphs/wine_[{'_'.join(map(str, seq))}][depth{d}]-{round(accuracy * 100, 1)}",
    format="png", cleanup=True)




