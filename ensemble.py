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
##########################
# test train split
X_train, X_test, label_train, label_test = train_test_split(train_df.drop(columns=['AdoptionSpeed', 'PetID']),
                                                            train_df['AdoptionSpeed'],
                                                            test_size=0.3, random_state=41)

# standard scaler (omit for random forest)
std = StandardScaler()
std.fit(X_train)
X_train = std.transform(X_train)
X_test = std.transform(X_test)
##########################
# option 1 ###################
# one hot encode the dummy variables stored as values
train_df2 = pd.get_dummies(train_df, columns=['Type', 'Breed1', 'Breed2', 'Gender', 'Color1',
                                              'Color2', 'Color3', 'Vaccinated', 'Dewormed',
                                              'Sterilized', 'Health', 'State'])

# test train split
X2_train, X2_test, label_train, label_test = train_test_split(train_df2.drop(columns=['AdoptionSpeed', 'PetID']),
                                                            train_df2['AdoptionSpeed'],
                                                            test_size=0.3, random_state=41)

# standard scaler (omit for random forest)
std = StandardScaler()
std.fit(X2_train)
X2_train = std.transform(X2_train)
X2_test = std.transform(X2_test)

# option 2 ##################
# balanced class weights (with frequency encoding for breed categories)
freq = train_df.groupby(['Breed1']).size().sort_values(ascending=False)
freq = freq.reset_index()
freq.columns = ['Breed1', 'Count']
freq_dict = dict(zip(freq['Breed1'], freq['Count']))

train_df_freq = train_df.copy()
train_df_freq['Breed1'] = train_df_freq['Breed1'].replace(freq_dict)
train_df_freq['Breed2'] = train_df_freq['Breed2'].replace(freq_dict)

# test train split
Xf_train, Xf_test, label_train, label_test = train_test_split(train_df_freq.drop(columns=['AdoptionSpeed', 'PetID']),
                                                            train_df_freq['AdoptionSpeed'],
                                                            test_size=0.3, random_state=41)

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
    curr_model.fit(Xf_train, label_train)
    with open(f"models/{model_name}.pickle", "wb") as pfile:
        pickle.dump(curr_model, pfile)

# ensembling from pre-trained models ###########
# Load pre-trained/tuned models
models = ['logit', 'rf', 'et', 'adb', 'svm']
for model_name in models:
    with open(f"models/{model_name}.pickle", "rb") as pfile:
        exec(f"{model_name} = pickle.load(pfile)")

model_vars = [eval(n) for n in models]
model_list = list(zip(models, model_vars))

for model_name in models:
    curr_model = eval(model_name)
    print(f'{model_name} score: {f1_score(label_test, curr_model.predict(Xf_test))}')

# knn score: 0.3842794759825327
# logit score: 0.49096022498995573
# rf score: 0.5743639921722113
# et score: 0.5353053435114504
# adb score: 0.5383211678832116
# svm score: 0.5563347358578775

# scores with one-hot encoding of categorical features
# logit score: 0.5140311804008909
# rf score: 0.5264586160108548
# et score: 0.5053813757604119
# adb score: 0.549127640036731
# svm score: 0.47935548841893255

# scores with frequency encoded breed features
# logit score: 0.48986083499005967
# rf score: 0.568
# et score: 0.5406721870433512
# adb score: 0.5387453874538746
# svm score: 0.45072115384615385

# create meta-classifier
stacked = StackingClassifier(classifiers=model_vars,
                             meta_classifier=BernoulliNB(),
                             use_probas=False)
stacked.fit(X2_train, label_train)

print(f'stacked score: {f1_score(label_test, stacked.predict(X2_test))}')
# stacked score: 0.5604166666666668 (Bernoulli meta)
# stacked score: 0.5366666666666667 (Logistic meta)
# stacked score: 0.5636101776284205 (Logistic meta with balanced class weights)
# stacked score: 0.5480459770114943 (Logistic meta with balanced class weights and one-hot categories)
# stacked score: 0.533467539003523 (Bernoulli meta with one-hot categories)
