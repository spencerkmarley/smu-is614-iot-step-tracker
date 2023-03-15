import pickle
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

digits = load_digits()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

logisticRegr = LogisticRegression(max_iter=10000)
logisticRegr.fit(x_train, y_train)

filename = 'model.sav'
pickle.dump(logisticRegr, open(filename, 'wb'))
