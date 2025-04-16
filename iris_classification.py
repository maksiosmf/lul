import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

sns.set(style="whitegrid")
plt.figure(figsize=(12, 10))

df = pd.read_csv('iris (1).csv')

print("Dataset Information:")
print(df.info())
print("\nSample data:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.countplot(x='species', data=df, palette='viridis')
plt.title('Distribution of Iris Species')

plt.figure(figsize=(12, 10))
sns.pairplot(df, hue='species', palette='viridis')
plt.suptitle('Pairplot of Iris Dataset Features', y=1.02)
plt.savefig('iris_pairplot.png')

plt.figure(figsize=(10, 8))
correlation = df.drop('species', axis=1).corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('iris_correlation.png')

X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

classifiers = [
    ('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors=3)),
    ('Linear SVM', SVC(kernel='linear', C=0.025, random_state=42)),
    ('RBF SVM', SVC(gamma=2, C=1, random_state=42)),
    ('Decision Tree', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('Random Forest', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42)),
    ('Neural Net', MLPClassifier(alpha=1, max_iter=1000, random_state=42)),
    ('AdaBoost', AdaBoostClassifier(random_state=42)),
    ('Naive Bayes', GaussianNB())
]

results = {}
cv_results = {}

for name, clf in classifiers:
    pipeline = make_pipeline(StandardScaler(), clf)
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    cv_results[name] = cv_scores.mean()
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cross-validation average score: {cv_scores.mean():.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')

plt.figure(figsize=(12, 6))
models = list(results.keys())
accuracy = list(results.values())
cv_accuracy = list(cv_results.values())

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - width/2, accuracy, width, label='Test Accuracy')
rects2 = ax.bar(x + width/2, cv_accuracy, width, label='CV Accuracy')

ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by Model')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig('model_comparison.png')

print("\nModel Ranking by Test Accuracy:")
for i, (model, acc) in enumerate(sorted(results.items(), key=lambda x: x[1], reverse=True), 1):
    print(f"{i}. {model}: {acc:.4f}")

print("\nModel Ranking by Cross-Validation Accuracy:")
for i, (model, acc) in enumerate(sorted(cv_results.items(), key=lambda x: x[1], reverse=True), 1):
    print(f"{i}. {model}: {acc:.4f}")

plt.show() 