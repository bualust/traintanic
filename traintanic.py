from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sklearn.metrics as sklm
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import category_encoders as ce

#training data
trainDF  = pd.read_csv('train.csv', header=0, index_col=None)
num_feat = trainDF.select_dtypes([np.number]).columns
num_feat = num_feat.drop('Survived')
cat_feat = trainDF.select_dtypes(exclude=[np.number]).columns
inp_feat = num_feat.append(cat_feat)

# fit model
bst = XGBClassifier()
bst.set_params(eval_metric=['error', 'logloss','auc'],
               max_depth=2, early_stopping_rounds=10)

le = LabelEncoder()
trainDF[cat_feat] = trainDF[cat_feat].apply(lambda x: le.fit_transform(x))

X, X_test, Y, Y_test = train_test_split(
                                trainDF[inp_feat],
                                trainDF['Survived'],
                                test_size=0.33, random_state=5)

eval_set = [(X,Y),(X_test, Y_test)]
bst.fit(X,Y,eval_set=eval_set,verbose=False)
print(bst)

print('Feature importance', num_feat, bst.feature_importances_)

y_pred = bst.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

results = bst.evals_result()
epochs = len(results["validation_0"]["error"])
x_axis = range(0, epochs)

# plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results["validation_0"]["logloss"], label="Train")
ax.plot(x_axis, results["validation_1"]["logloss"], label="Test")
ax.legend()
plt.ylabel("Log Loss")
plt.xlabel("Iteration")
plt.title("XGBoost Log Loss")
plt.show()

# plot classification error
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('Classification Error')
plt.xlabel("Iteration")
plt.title('XGBoost Classification Error')
plt.show()

# plot classification auc
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['auc'], label='Train')
ax.plot(x_axis, results['validation_1']['auc'], label='Test')
ax.legend()
plt.ylabel('AUC')
plt.xlabel("Iteration")
plt.title('XGBoost Training Performance')
plt.show()

# plot ROC curve
y_pred_proba_train = bst.predict_proba(X)
y_pred_proba_test  = bst.predict_proba(X_test)


fpr_train, tpr_train, _ = sklm.roc_curve(Y, y_pred_proba_train[:,1])
fpr_test, tpr_test, _   = sklm.roc_curve(Y_test, y_pred_proba_test[:,1])

auc_train = sklm.auc(fpr_train, tpr_train)
auc_test  = sklm.auc(fpr_test, tpr_test)

fig, ax = plt.subplots()
plt.title(f"ROC curve, AUC=(test: {auc_test:.4f}, train: {auc_train:.4f})")
plt.plot(fpr_test, tpr_test, label="test data")
plt.plot(fpr_train, tpr_train, label="train data")
ax.legend()
plt.ylabel('ROC Curve')
plt.show()

##test data
#testDF   = pd.read_csv('test.csv', header=0, index_col=None)
