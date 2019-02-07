import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
np.set_printoptions(suppress=True) # Suppress scientific notation where possible

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score,f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import cohen_kappa_score
import pickle

# data pre-processing #########################
# training data
train_df = pd.read_csv('data/train.csv')

# drop a few columns and restrict to quantity == 1
train_df = train_df[train_df['Quantity']==1].drop(columns=['Description', 'VideoAmt', 'RescuerID', 'Name', 'Quantity'])
train_sent = pd.read_csv('sent_train.csv', index_col=0)

# keep only the records that have sentiment score
train_df = train_df.merge(train_sent, how='inner', left_on='PetID', right_index=True)
train_df['AdoptionSpeed'] = train_df['AdoptionSpeed'].replace({1: 0, 2: 0, 3: 0, 4: 1})

# test train split
X_train, X_test, label_train, label_test = train_test_split(train_df.drop(columns=['AdoptionSpeed', 'PetID']),
                                                            train_df['AdoptionSpeed'],
                                                            test_size=0.3, random_state=41)

# standard scaler (omit for random forest)
std = StandardScaler()
std.fit(X_train)
X_train = std.transform(X_train)
X_test = std.transform(X_test)

# ensemble models
logit = LogisticRegression(C=0.95, class_weight='balanced')
rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1,
                            class_weight='balanced')

# extremely randomized trees (a random subset of features is used as in RF,
# additionally, instead of the most discriminative thresholds, randomly generated
# thresholds are used as splitting rules
et = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1,
                            class_weight='balanced')

# in AdaBoost, weights of incorrectly classified instances are adjusted so
# the model will focusing more on these difficult cases
adb = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3, class_weight='balanced'),
                         n_estimators=100)

svm = SVC(C=0.95, class_weight='balanced')

models = ['logit', 'rf', 'et', 'adb', 'svm']

for model_name in models:
    curr_model = eval(model_name)
    curr_model.fit(X_train, label_train)
    with open(f"models/{model_name}.pickle", "wb") as pfile:
        pickle.dump(curr_model, pfile)

# ensembling from pre-trained models ###########
# Load pre-trained/tuned models
models = ['rf', 'et', 'adb', 'svm']
for model_name in models:
    with open(f"models/{model_name}.pickle", "rb") as pfile:
        exec(f"{model_name} = pickle.load(pfile)")

model_vars = [eval(n) for n in models]
model_list = list(zip(models, model_vars))

for model_name in models:
    curr_model = eval(model_name)
    print(f'{model_name} score: {f1_score(label_test, curr_model.predict(X_test))}')

# knn score: 0.3842794759825327
# logit score: 0.49096022498995573
# rf score: 0.5743639921722113
# et score: 0.5353053435114504
# adb score: 0.5383211678832116
# svm score: 0.5563347358578775

# create meta-classifier
stacked = StackingClassifier(classifiers=model_vars,
                             meta_classifier=LogisticRegression(class_weight='balanced'),
                             use_probas=False)
stacked.fit(X_train, label_train)

print(f'stacked score: {f1_score(label_test, stacked.predict(X_test))}')
# stacked score: 0.5604166666666668 (Bernoulli meta)
# stacked score: 0.5366666666666667 (Logistic meta)
# stacked score: 0.5636101776284205 (Logistic meta with balanced class weights)

