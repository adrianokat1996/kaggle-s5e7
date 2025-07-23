import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train = pd.read_csv('datas/train.csv')
test = pd.read_csv('datas/test.csv')

# Basic info
print('Train shape:', train.shape)
print('Test shape:', test.shape)
print('\nTrain columns:', train.columns.tolist())
print('\nTrain head:')
print(train.head())
print('\nTest head:')
print(test.head())

# Missing values
print('\nMissing values in train:')
print(train.isnull().sum())
print('\nMissing values in test:')
print(test.isnull().sum())

# Describe features
print('\nTrain describe:')
print(train.describe())
print('\nTest describe:')
print(test.describe())

# Visualize feature distributions
features = [col for col in train.columns if col not in ['target']]
for col in features:
    plt.figure(figsize=(10, 4))
    sns.histplot(train[col], kde=True, color='blue', label='Train', stat='density')
    sns.histplot(test[col], kde=True, color='orange', label='Test', stat='density')
    plt.title(f'Distribution of {col} (Train vs Test)')
    plt.legend()
    plt.show()

# Target distribution (if available)
if 'target' in train.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=train)
    plt.title('Target Distribution in Train Set')
    plt.show()
