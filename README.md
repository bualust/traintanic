# TrainTanic
An XGBoost binary classifier to give a survival probability to the Titanic disaster.

A training dataset is used from the [Kaggle competition](https://www.kaggle.com/competitions/titanic)

The model is trained over 67% of the training data and its accuracy is evaultaed on the remaining 33% of the events to be roughly 83%.

The AUC is assed to be around 87-90%.

## to run
This simple script requires the following dependencies: xgboost, pandas, matplotlib, sklearn
> python3 traintanic.py
