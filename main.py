# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample data: [weight (g), color (0=green, 1=red), size (cm)]
# 0=apple, 1=banana, 2=orange
data = np.array([
    [150, 1, 7],  # apple
    [170, 1, 8],  # apple
    [140, 1, 6],  # apple
    [130, 0, 12], # banana
    [160, 0, 13], # banana
    [145, 0, 11], # banana
    [180, 1, 8],  # apple
    [120, 0, 14], # banana
    [190, 1, 7],  # apple
    [155, 0, 9],  # orange
    [165, 0, 8],  # orange
    [175, 0, 7]   # orange
])

# Labels: 0=apple, 1=banana, 2=orange
labels = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 2, 2, 2])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Test the classifier with a new fruit
new_fruit = np.array([[155, 1, 9]])  # Example: a fruit with weight 155g, red color, and 9cm size
prediction = clf.predict(new_fruit)

fruit_names = ['apple', 'banana', 'orange']
print(f"The new fruit is classified as: {fruit_names[prediction[0]]}")

# Test the classifier with a another new fruit
new_fruit = np.array([[175, 0, 9]])  # Example: a fruit with weight 175g, orange color, and 9cm size
prediction = clf.predict(new_fruit)

fruit_names = ['apple', 'banana', 'orange']
print(f"The new fruit is classified as: {fruit_names[prediction[0]]}")

# Test the classifier with a another new fruit
new_fruit = np.array([[135, 0, 12]])  # Example: a fruit with weight 135g, yellow color, and 12cm size
prediction = clf.predict(new_fruit)

fruit_names = ['apple', 'banana', 'orange']
print(f"The new fruit is classified as: {fruit_names[prediction[0]]}")
