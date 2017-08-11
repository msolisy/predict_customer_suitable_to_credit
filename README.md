# Predict customers that are suitable to credit

## Instructions to run the code:

The script was written in Python 3 and using the next libraries: 
* Python 3.6.1
* Pandas 0.20.1
* Numpy 1.12.1
* Scikit-learn 0.18.1

To execute by terminal *it's necessary two files*:
* puzzle_train_dataset.csv
* puzzle_test_dataset.csv

And then simply run:
```
python3 predict.py
```

After the execution will be created the file *predictions.csv* 

```
Start
Total size: 54939
Train size:  43961
Validate size:  10978
Model name:  Multinomial Naive Bayes , logLoss:  13.6830160162
Model name:  Naive Bayes , logLoss:  6.69515523012
Model name:  Random Forest Classifier , logLoss:  5.0433454606
Model name:  Nearest Neighbors , logLoss:  6.30811566291
Model name:  AdaBoost , logLoss:  4.79794864987
Model name:  Linear SVM , logLoss:  24.1254574611
Model name:  Dummy , logLoss:  5.59705622213

Model with best result:  AdaBoost , Min LogLoss:  4.79794864987
File created:  predictions.csv
End
```
