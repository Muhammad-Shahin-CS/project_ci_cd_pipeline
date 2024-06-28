import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('dataset.csv')

# Encode categorical variables
label_encoder = LabelEncoder()
df['color'] = label_encoder.fit_transform(df['color'])
df['movement'] = label_encoder.fit_transform(df['movement'])

# Prepare data for training
X = df[['legs', 'color', 'movement']]
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a decision tree classifier
model = DecisionTreeClassifier(random_state=42)

# Train the classifier
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Example prediction
example_data = pd.DataFrame([[2, 1, 0]], columns=['legs', 'color', 'movement'])  # Example: [legs=2, color=1 (blue), movement=0 (fly)]
example_prediction = model.predict(example_data)
print(f"Prediction result: {'bird' if example_prediction[0] == 'bird' else 'cat'}")
