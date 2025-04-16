import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

df = pd.read_csv('iris (1).csv')

X = df[['petal_length', 'petal_width']].values
y = df['species'].values

species_mapping = {species: i for i, species in enumerate(df['species'].unique())}
y_numeric = np.array([species_mapping[species] for species in y])

X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.3, random_state=42)

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "Naive Bayes",
    "QDA"
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

figure = plt.figure(figsize=(20, 10))
i = 1

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])

h = 0.02

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

ax = plt.subplot(2, 4, i)
ax.set_title("Input data")

scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xticks(())
ax.set_yticks(())

legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower right", title="Classes")
ax.add_artist(legend1)

i += 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

for name, clf in zip(names, classifiers):
    ax = plt.subplot(2, 4, i)
    
    pipeline = make_pipeline(StandardScaler(), clf)
    
    pipeline.fit(X_train, y_train)
    
    score = pipeline.score(X_test, y_test)
    
    DecisionBoundaryDisplay.from_estimator(
        pipeline, X, cmap=plt.cm.RdYlBu, alpha=0.8, ax=ax, eps=0.5
    )
    
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(x_max - 0.3, y_min + 0.3, ('%.2f' % score).lstrip('0'), size=15,
            horizontalalignment='right')
    
    i += 1

plt.tight_layout()
plt.savefig('iris_decision_boundaries.png', dpi=300)
plt.show()

from sklearn.decomposition import PCA

print("\nCreating PCA visualization for all features...")

X_all = df.drop('species', axis=1).values
y_all = df['species'].values

y_all_numeric = np.array([species_mapping[species] for species in y_all])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_all)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y_all_numeric, test_size=0.3, random_state=42
)

x_min_pca, x_max_pca = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
y_min_pca, y_max_pca = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5

figure_pca = plt.figure(figsize=(20, 10))
i = 1

ax = plt.subplot(2, 4, i)
ax.set_title("PCA-transformed data")

scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_pca, cmap=cm_bright, edgecolors='k')
ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_pca, cmap=cm_bright, alpha=0.6, edgecolors='k')
ax.set_xlim(x_min_pca, x_max_pca)
ax.set_ylim(y_min_pca, y_max_pca)
ax.set_xticks(())
ax.set_yticks(())

legend2 = ax.legend(*scatter.legend_elements(),
                    loc="lower right", title="Classes")
ax.add_artist(legend2)

i += 1

xx_pca, yy_pca = np.meshgrid(
    np.arange(x_min_pca, x_max_pca, h),
    np.arange(y_min_pca, y_max_pca, h)
)

for name, clf in zip(names, classifiers):
    ax = plt.subplot(2, 4, i)
    
    pipeline = make_pipeline(StandardScaler(), clf)
    
    pipeline.fit(X_train_pca, y_train_pca)
    
    score = pipeline.score(X_test_pca, y_test_pca)
    
    DecisionBoundaryDisplay.from_estimator(
        pipeline, X_pca, cmap=plt.cm.RdYlBu, alpha=0.8, ax=ax, eps=0.5
    )
    
    ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_pca, cmap=cm_bright, edgecolors='k')
    ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test_pca, cmap=cm_bright, alpha=0.6, edgecolors='k')
    
    ax.set_xlim(x_min_pca, x_max_pca)
    ax.set_ylim(y_min_pca, y_max_pca)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(x_max_pca - 0.3, y_min_pca + 0.3, ('%.2f' % score).lstrip('0'), size=15,
            horizontalalignment='right')
    
    i += 1

plt.tight_layout()
plt.savefig('iris_pca_decision_boundaries.png', dpi=300)
plt.show()

print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

print("Decision boundary visualization complete. Check the output files.") 