# Importing libraries and functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def showPiePlot(n, m):
    """
        Function for plotting pie charts for features m to n
        m - n must be 6 as the plot has 6 subplots
    """
    subplotNumber = 1
    plt.figure(figsize=(14, 10))
    for i in range(n, m):
        feature_counts = df[obj_features[i]].value_counts()
        counts = []
        labels = []

        for x, y in feature_counts.items():
            counts.append(y)
            labels.append(x)
        print("COunts*******")
        print(counts)
        print ("LABELS----------")
        print(labels)
        plt.subplot(2, 3, subplotNumber)
        subplotNumber += 1
        plt.xlabel(obj_features[i])
        plt.pie(counts, labels=labels)

    plt.show()


def showBarPlot(m, n):
    """
        Arrange and Display data in bar plots from feature m to n
        m - n must be 6 as the plot has 6 subplots
    """
    subplotNumber = 1
    plt.figure(figsize=(14, 10))
    for k in range(m, n):

        feature_data = df[obj_features[k]]
        groups = np.unique(feature_data)
        b = {g: [0, 0] for g in groups}

        # Get Count of poisonous and edible mushrooms for each value of the feature
        for i in range(len(feature_data)):
            if df['class'][i] == 'p':
                b[feature_data[i]][0] += 1
            else:
                b[feature_data[i]][1] += 1

        # Arrange Count of poisonous and edible mushrooms for each value of the feature
        # to be in two lists P(poisonous) and E(edible)
        P = []
        E = []
        for i in b.values():
            P.append(i[0])
            E.append(i[1])
        print(P)
        print(E)

        X_axis = np.arange(len(groups))
        print("At k = ", k, "", k % n)

        plt.subplot(2, 3, subplotNumber)
        subplotNumber += 1
        plt.xticks(X_axis, groups)
        plt.bar(X_axis - 0.2, P, width=0.4, label='Poisonsous')
        plt.bar(X_axis + 0.2, E, width=0.4, label='Edible')
        plt.xlabel("Values for " + obj_features[k])
        plt.ylabel("Number of Mushrooms")
        plt.title("Number of Mushrooms in each group")
        plt.legend()

    plt.show()


def FillMissingData(df, obj_features):
    df_obj = df[obj_features]
    si = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    si.fit(df_obj)
    df_obj = si.transform(df_obj)

    df[obj_features] = df_obj
    return df


# Reading data from csv file


df = pd.read_csv('mushrooms 2.csv')
print(df.head())

# Printing shape of the dataset


print(df.shape)

# Getting numerical features and categorical features


num_features = df.select_dtypes(exclude=['object']).columns.tolist()
obj_features = df.select_dtypes(include=['object']).columns.tolist()

# Filling missing data


df = FillMissingData(df, obj_features)
print(df.head())

# Plotting bar plots of each feature value with respect to count of poisonous or non-poisonous


showBarPlot(1, 7)

showBarPlot(7, 13)

showBarPlot(13, 19)

showBarPlot(19, 23)

# Plotting Pie charts for each feature with all its possible values count


showPiePlot(1, 7)

showPiePlot(7, 13)

showPiePlot(13, 19)

showPiePlot(19, 23)

# Labeling the categorical data


labelEncoder = LabelEncoder()
map = []
for c in obj_features:
    labelEncoder.fit(df[c])
    df[c] = labelEncoder.transform(df[c])
    dic = {index: label for index, label in enumerate(labelEncoder.classes_)}
    map.append(dic)
print(df.head())

# Separating input(X) from output(Y)


Y = df['class']
X = df.drop('class', axis=1)
print(X.head())

# Normalizing the input X using MinMaxScaler


scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print(X.head())

# Plotting Kendall Correlation Matrix


corr = X.corr(method='kendall')
plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True)
plt.show()

# Plotting Pearson Correlation Matrix


corr = X.corr(method='pearson')

plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True)
plt.show()

# We can see here that veil-type has only 1 value so there is no need for it. As it has no contribution in classifying data.
# So we can drop this feature.
# veil-color feature is also strongly correlated to gill-attachment with r= 0.9.
# So we can drop any of these two features.
#


print(X['veil-type'].any())
print("*************************************************")
X1 = X.drop('veil-type', axis=1)
#
X1 = X1.drop('veil-color', axis=1)

X1 = X1.drop('bruises', axis=1)

obj_features.remove('class')
obj_features.remove('veil-type')
obj_features.remove('veil-color')
obj_features.remove('bruises')

# Split Data into training and testing sets


X_train, X_test, Y_train, Y_test = train_test_split(X1, Y, train_size=0.8, random_state=0)

# Importing functions to measure the accuracy of each model


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# ### Logistic Regression


logisiticRegression = LogisticRegression(random_state=100)

logisiticRegression.fit(X_train, Y_train)

y_pred = logisiticRegression.predict(X_test)

print("Linear Regression:\n")
print("Accuracy: ", accuracy_score(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))

# ### Gaussian Naive Bayes


from sklearn.naive_bayes import GaussianNB

GNB = GaussianNB()

GNB.fit(X_train, Y_train)

y_pred = GNB.predict(X_test)

print("Gaussian Naive Bayes:\n")
print("Accuracy: ", accuracy_score(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))

# ### K-Nearest Neighbors


from sklearn.neighbors import KNeighborsClassifier

print("KNN------")
for i in range(1, 6):
    KNN = KNeighborsClassifier(n_neighbors=i)

    KNN.fit(X_train, Y_train)

    y_pred = KNN.predict(X_test)

    print(f"Accuracy with k={i}: ", accuracy_score(Y_test, y_pred))
    print(f"Confusion Matrix with k={i}:\n", confusion_matrix(Y_test, y_pred))

# ### Decision Tree

print("Decision Tree")
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

DT = DecisionTreeClassifier()

DT = DT.fit(X_train, Y_train)

y_pred = DT.predict(X_test)

print("Accuracy: ", accuracy_score(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))

# Plotting Decision Tree


plt.figure(figsize=(16, 16))
tree.plot_tree(DT, feature_names=obj_features)
plt.show()