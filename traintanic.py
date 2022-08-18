from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import category_encoders as ce

#training data
trainDF  = pd.read_csv('train.csv', header=0, index_col=None)
num_feat = trainDF.select_dtypes([np.number]).columns
num_feat = num_feat.drop('Survived')
cat_feat = trainDF.select_dtypes(exclude=[np.number]).columns

# fit model
bst = XGBClassifier(n_estimators=2, max_depth=2,#early_stopping_rounds=5,
                    learning_rate=1, objective='binary:logistic')

X, X_test, Y, Y_test = train_test_split(
                                trainDF[num_feat],
                                trainDF['Survived'],
                                test_size=0.33, random_state=7)

#ce_bin = ce.BinaryEncoder(cols=trainDF['Sex'])
#print(ce_bin)
#ce_bin.fit_transform(X,Y)
#print(ce_bin)
bst.fit(X,Y)

y_pred = bst.predict(X_test)
predictions = [round(value) for value in y_pred]
print(predictions)
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


#test data
testDF   = pd.read_csv('test.csv', header=0, index_col=None)
