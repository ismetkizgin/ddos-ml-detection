from sklearn.tree import DecisionTreeClassifier #import Decision tree classifier
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

nRowsRead = None
train_data = pd.read_csv('./data/train.csv', delimiter=',', nrows = nRowsRead)
train_data.dataframeName = 'train.csv'
nRow, nCol = train_data.shape
print(f'There are {nRow} rows and {nCol} columns')

nRowsRead = None
test_data = pd.read_csv('./data/test.csv', delimiter=',', nrows = nRowsRead)
test_data.dataframeName = 'test.csv'
nRow, nCol = test_data.shape
print(f'There are {nRow} rows and {nCol} columns')

X_train = train_data.drop('Label',axis=1)
X_test = test_data.drop('Label',axis=1)
y_train = train_data['Label']
y_test = test_data['Label']

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
dt_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, dt_pred))