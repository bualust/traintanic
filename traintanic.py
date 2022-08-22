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
import csv
import math


def main():

    #training data
    trainDF  = pd.read_csv('train.csv', header=0, index_col=None)

    #list of input features
    num_feat, cat_feat = get_features(trainDF)
    inp_feat = num_feat.append(cat_feat)

    #fit model
    bst = XGBClassifier()
    bst.set_params(eval_metric=['error', 'logloss','auc'],
                   max_depth=2, early_stopping_rounds=10)

    #Encode string input variables
    le = LabelEncoder()
    trainDF[cat_feat] = trainDF[cat_feat].apply(lambda x: le.fit_transform(x))

    #fix vars with nan values
    trainDF = fixNaN(trainDF, inp_feat)

    #adding secondary features
    trainDF  = add_features(trainDF)
    num_feat, cat_feat_dum = get_features(trainDF)
    inp_feat = num_feat


    #plot features bf training
    plot_features(trainDF,inp_feat)

    #split labeled sample in train and test
    X, X_test, Y, Y_test = train_test_split(
                                    trainDF[inp_feat],
                                    trainDF['Survived'],
                                    test_size=0.10, random_state=5)

    eval_set = [(X,Y),(X_test, Y_test)]
    bst.fit(X,Y,eval_set=eval_set,verbose=False)
    bst.save_model('traintanic.json')

    get_feature_ranking(inp_feat,bst)
    get_accuracy(bst,X_test,Y_test)
    get_perf_plots(bst,X,Y,X_test, Y_test)
    get_survival_probabilities_lab(bst, trainDF, inp_feat)
    get_survival_probabilities_unlab(bst, inp_feat, cat_feat,le)


#fix variables with NaN values
def fixNaN(trainDF, num_feat):

    for feat_name in num_feat:
        new_values = []
        for values in trainDF[feat_name]:
            if math.isnan(values) :
                new_values.append(-99)
            else: new_values.append(values)
        trainDF[feat_name] = new_values

    return trainDF

#features list
def get_features(trainDF):

    #split categorical features from others
    num_feat = trainDF.select_dtypes([np.number]).columns
    num_feat = num_feat.drop('Survived')
    num_feat = num_feat.drop('PassengerId')
    cat_feat = trainDF.select_dtypes(exclude=[np.number]).columns
    cat_feat = cat_feat.drop('Name')

    return num_feat, cat_feat

#plot all input features
def plot_features(trainDF,inp_feat):
    fig, ax = plt.subplots()
    for ft in inp_feat:
        print('Plotting feature ', ft)
        trainDF_surv = trainDF[trainDF['Survived']>0]
        trainDF_nsur = trainDF[trainDF['Survived']<1]
        plt.hist(trainDF_surv[ft], histtype='step', label="survived")
        plt.hist(trainDF_nsur[ft], histtype='step', label="not survived")
        ax.legend()
        plt.xlabel(ft)
        plt.savefig('model_figs/'+ft+'.png')
        plt.cla()


#derive secondary features
def add_features(trainDF):
    trainDF['ParchSibSp_over_AgeClass']=trainDF['Parch']*trainDF['SibSp']/trainDF['Pclass']*trainDF['Age']
    trainDF['ParchClass']=trainDF['Parch']/trainDF['Pclass']
    trainDF['ParchSib']=(trainDF['Parch']-trainDF['SibSp'])/(trainDF['Parch']+trainDF['SibSp'])
    trainDF['AgeClass']=(trainDF['Age']-trainDF['Pclass'])/(trainDF['Age']+trainDF['Pclass'])
    trainDF['CabinClass']=(trainDF['Cabin']-trainDF['Pclass'])/(trainDF['Cabin']+trainDF['Pclass'])
    trainDF['CabinAge']=(trainDF['Cabin']-trainDF['Age'])/(trainDF['Cabin']+trainDF['Age'])
    trainDF['CabinSex']=(trainDF['Cabin']-trainDF['Sex'])/(trainDF['Cabin']+trainDF['Sex'])
    trainDF['Age_over_class']=trainDF['Age']/trainDF['Pclass']
    trainDF['Cabin_over_class']=trainDF['Cabin']/trainDF['Pclass']
    trainDF['ciao']=(trainDF['Cabin']/trainDF['Pclass']+trainDF['Age']/trainDF['Pclass'])
    return trainDF

#features importance ranking
def get_feature_ranking(inp_feat,bst):
    features_map = {}
    for name,imp in zip(inp_feat, bst.feature_importances_):
        features_map[name]=float(imp)
    features_map = dict(sorted(features_map.items(), key=lambda item:item[1],
                        reverse=True))
    print('Features importance ', features_map)
    return features_map

#Calculate accuracy
def get_accuracy(bst,X_test,Y_test):
    y_pred = bst.predict(X_test)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(Y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return 0

#Performance evaluation
def get_perf_plots(bst,X,Y,X_test, Y_test):
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
    plt.savefig('model_figs/logloss.png')

    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    plt.ylabel('Classification Error')
    plt.xlabel("Iteration")
    plt.title('XGBoost Classification Error')
    plt.savefig('model_figs/error.png')

    # plot classification auc
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['auc'], label='Train')
    ax.plot(x_axis, results['validation_1']['auc'], label='Test')
    ax.legend()
    plt.ylabel('AUC')
    plt.xlabel("Iteration")
    plt.title('XGBoost Training Performance')
    plt.savefig('model_figs/auc.png')

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
    plt.savefig('model_figs/roc.png')
    return 0

#survival probabilities
def get_survival_probabilities_lab(bst, trainDF, inp_feat):
    trainDF_alive = trainDF[trainDF['Survived']>0]
    trainDF_dead  = trainDF[trainDF['Survived']<1]
    alive_pred_proba = bst.predict_proba(trainDF_alive[inp_feat])
    dead_pred_proba = bst.predict_proba(trainDF_dead[inp_feat])

    fig, ax = plt.subplots()
    plt.title('Training sample')
    plt.hist(alive_pred_proba[:,1], histtype='step', label="true alive")
    plt.hist(dead_pred_proba[:,1], histtype='step', label="true dead")
    ax.legend()
    plt.xlabel('Survival Probability')
    plt.savefig('model_figs/labeled_prob.png')
    return 0

##unlabeled data
def get_survival_probabilities_unlab(bst, inp_feat, cat_feat,le):

    testDF   = pd.read_csv('test.csv', header=0, index_col=None)
    testDF[cat_feat] = testDF[cat_feat].apply(lambda x: le.fit_transform(x))
    testDF   = add_features(testDF)

    y_pred = bst.predict(testDF[inp_feat])
    with open('results.csv', 'w', newline='') as file:
        fieldnames = ['PassengerId', 'Survived']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for psgID, pred in zip(testDF['PassengerId'], y_pred):
            writer.writerow({'PassengerId': int(psgID),
                             'Survived': int(pred)})
    predictions = [round(value) for value in y_pred]
    test_proba = bst.predict_proba(testDF[inp_feat])

    fig, ax = plt.subplots()
    plt.title('Unlabeled sample')
    plt.hist(test_proba[:,1], histtype='step', label="Unlabeled")
    ax.legend()
    plt.xlabel('Survival Probability')
    plt.savefig('model_figs/unlabeled_prob.png')
    return 0

#main
if __name__ == '__main__':
    main()
