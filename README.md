# pet_clf

Using different classification methods (logistic regression, random forests, xgboost (gradient-boosted trees)), I tried to classify whether pets at petfinder.my will be adopted within a 100 day time window. Data from Kaggle competition was used for the analysis.

Features that were discovered as important include age, breed, sentiment scores of animal profile description, etc. I was able to progressively get better f1 scores after treating for unbalanced sample sizes and selecting the appropriate model configuration after cross-validating hyper-parameters.

![Alt text](/pet_clf_models.png?raw=true)
![Alt text](/pet_clf_findings.png?raw=true)
