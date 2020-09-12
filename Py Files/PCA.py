#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Sources: https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
# https://www.kaggle.com/shrutimechlearn/step-by-step-pca-with-iris-dataset

from pandas import read_csv
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing 

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Species']
iris_dataset = read_csv(url, names = names)

print (iris_dataset.head())

# Since class names in Species column are of type string; Encoding it to integer values
label_encoder = preprocessing.LabelEncoder() 
iris_dataset['Species']= label_encoder.fit_transform(iris_dataset['Species']) 

iris_dataset['Species'].unique()


# In[ ]:


# ~~~~~~~~~ PCA as dimensionality reduction ~~~~~~~~~~~

import warnings

warnings.simplefilter("ignore")

Y = iris_dataset.Species
X = iris_dataset.drop(['Species'],axis=1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=20, stratify=Y)

pca = PCA()
X_new = pca.fit_transform(X)

pca.get_covariance()

# explained variance ratio by a principal component = 
# the ratio between the variance of that principal component and the total variance
explained_variance = pca.explained_variance_ratio_
print ("Explained Variance: ", explained_variance, "\n")

# How to choose number of components?
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

#
with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(4), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()


# In[ ]:


# Training KN classifier over reduced features 
pca = PCA(n_components = 3)
X_new = pca.fit_transform(X)

X_train_new, X_test_new, y_train, y_test = train_test_split(X_new, Y, test_size = 0.3, random_state=20, stratify=Y)

knn_pca = KNeighborsClassifier(7)
knn_pca.fit(X_train_new,y_train)
print("Train score after PCA: ", knn_pca.score(X_train_new, y_train),"%")
print("Test score after PCA: ", knn_pca.score(X_test_new, y_test),"%")
print ("Explained Variance: ", pca.explained_variance_ratio_, "\n")

# Visualising the Test set results
classifier = knn_pca
X_set, y_set = X_test_new, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel(),np.zeros((X1.shape[0],X1.shape[1])).ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN PCA (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# In[ ]:


# ~~~~~~~~~ PCA for visualization ~~~~~~~~~~~

# Load multidimensional digits dataset
from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

# Num features = 64 (= 8*8 pixels)

pca = PCA(2)  # project from 64 to 2 dimensions
projected = pca.fit_transform(digits.data)
print("Original Shape: ", digits.data.shape)

print("PCA Reduced Shape: ", projected.shape)

plt.scatter(projected[:, 0], projected[:, 1],
            c = digits.target, edgecolor='none', alpha=0.5,
            cmap = plt.cm.Wistia) 
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();


# In[ ]:


# Now using MNIST Digits dataset
# How to choose number of components for Digits dataset?

pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[ ]:


# ~~~~~~~~~ PCA for noise filtering ~~~~~~~~~~~

def plot_digits(data, descriptn):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    print (descriptn)
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap='binary', interpolation='nearest',
                  clim=(0, 16))
        
# Original Digits
plot_digits(digits.data, "Original Digits:")

# Adding random noise to create a noisy dataset
np.random.seed(42)
noisy = np.random.normal(digits.data, 4)
plot_digits(noisy, "Noise Distorted Digits:")

# Using PCA
pca = PCA(0.50).fit(noisy)
print ("Number of components for projection preserve of 50% of variance: ", pca.n_components_)

components = pca.transform(noisy)
filtered = pca.inverse_transform(components)

plot_digits(filtered, "PCA Re-constructed Digits:")


# In[ ]:




