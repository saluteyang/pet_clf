import pandas as pd
import glob
import json

# sent_dict = dict()
# for f in glob.glob('data/test_sentiment/*.json'):
#     newf = f.replace('.', '/').split('/')
#     newf = [x for x in newf if x not in ['data', 'test_sentiment', 'json']][0]
#     with open(f) as json_data:
#         sent_dict[newf] = json.load(json_data)['documentSentiment']
#
# df = pd.DataFrame.from_dict(sent_dict)
# df = df.T
#
# df_test = df.copy()
# df_train = df.copy()
#
# df_test.to_csv('sent_test.csv', index=True)
# df_train.to_csv('sent_train.csv', index=True)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# plt.style.use('ggplot')
sns.set()
np.set_printoptions(suppress=True) # Suppress scientific notation where possible

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score,f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import export_graphviz, DecisionTreeClassifier

# data pre-processing #########################
# training data
train_df = pd.read_csv('data/train.csv')

# drop a few columns and restrict to quantity == 1
train_df = train_df[train_df['Quantity']==1].drop(columns=['Description', 'VideoAmt', 'RescuerID', 'Name', 'Quantity'])
train_sent = pd.read_csv('sent_train.csv', index_col=0)

# keep only the records that have sentiment score
train_df = train_df.merge(train_sent, how='inner', left_on='PetID', right_index=True)
train_df['AdoptionSpeed'] = train_df['AdoptionSpeed'].replace({1: 0, 2: 0, 3: 0, 4: 1})

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

# model fitting ###########################
# logistic CV with f1_score scoring
# logitCV = LogisticRegressionCV(cv=5, random_state=0, scoring='f1_macro').fit(X2_train, label_train)
# logitCV.score(X2_train, label_train)   # 0.77 (acc); 0.636 (acc, f1_macro scoring)
# logitCV.score(X2_test, label_test)   # 0.74 (acc); 0.598 (acc, f1_macro scoring)
#
# f1_score(label_train, logitCV.predict(X2_train))   # 0.41
# f1_score(label_test, logitCV.predict(X2_test))   # 0.36

# logistic with manually set regularization
logit = LogisticRegression(C=0.95)
logit.fit(X2_train, label_train)
print("The score for logistic regression is")
print("Training: {:.2f}%".format(100*f1_score(label_train, logit.predict(X2_train))))
print("Test set: {:.2f}%".format(100*f1_score(label_test, logit.predict(X2_test))))

# interpretation and diagnostics ###################
# select numerical columns for pairplot
numeric_cols = ['Age', 'MaturitySize', 'FurLength', 'Fee', 'PhotoAmt',
                'magnitude', 'score', 'AdoptionSpeed']

def plot_features(df):
    sample = (df[numeric_cols].sample(1000, random_state=44))
    sns.pairplot(sample, hue='AdoptionSpeed', plot_kws=dict(alpha=.3, edgecolor='none'))

plot_features(train_df)

# show coefs for logistic regression with feature names
feature_names = train_df2.drop(columns=['AdoptionSpeed', 'PetID']).columns
feature_coefs = pd.DataFrame({'feature': feature_names, 'coef': logit.coef_[0]})
feature_coefs['abs_coef'] = abs(feature_coefs['coef'])
feature_coefs.sort_values(by='abs_coef', ascending=False)

#           feature      coef  abs_coef
# 173    Breed1_307  0.601116  0.601116
# 0             Age  0.394190  0.394190
# 146    Breed1_276 -0.317996  0.317996  (Maine Coon, largest domestic breed)
# 100    Breed1_207 -0.313454  0.313454
# 4        PhotoAmt -0.294131  0.294131
# 8          Type_2  0.276278  0.276278
# 7          Type_1 -0.276278  0.276278
# 80     Breed1_169 -0.261142  0.261142
# 99     Breed1_206 -0.249684  0.249684
# 330  Sterilized_2 -0.242027  0.242027

# ad hoc exploratory analysis  #############
# breed count
train_df.groupby(['Type', 'Breed1'])[['Breed1']].count().sort_values(by='Breed1', ascending=False)[:5]

# 1    307       4610  mixed breed dog
# 2    266       2463  domestic short hair cat

# photo count histogram
grouped = train_df.groupby(['PhotoAmt'])['Type'].count().sort_values(ascending=False)
grouped.head()
train_df['PhotoAmt'].hist()
plt.show()

# Plot decision region (optional) ###########################
# plot_df2 = train_df2[['Age', 'PhotoAmt', 'AdoptionSpeed']]
#
# X_train_plot, X_test, label_train, label_test = train_test_split(plot_df2.drop(columns=['AdoptionSpeed']),
#                                                             plot_df2['AdoptionSpeed'],
#                                                             test_size=0.3, random_state=41)
#
#
# std = StandardScaler()
# std.fit(X_train_plot)
# X_train_plot = std.transform(X_train_plot)
#
# logit_plot = LogisticRegression(C=0.95)
# logit_plot.fit(X_train_plot, label_train)
#
# plot_decision_regions(X=X_train_plot,
#                       y=label_train.values,
#                       clf=logit_plot,
#                       legend=2,
#                       markers="o")
# plt.gcf().set_size_inches(12, 10)
# plt.show()

######################################
# how did the language score perform?
feature_coefs[feature_coefs['feature'].isin(['magnitude', 'score'])]

# Print confusion matrix for logistic regression
logit_confusion = confusion_matrix(label_test, logit.predict(X2_test))
plt.figure(dpi=150)
sns.heatmap(logit_confusion, cmap=plt.cm.Blues, annot=True, square=True,
           xticklabels=['adopted', 'not adopted'],
           yticklabels=['adopted', 'not adopted'])
plt.xlabel('Predicted Adoptions')
plt.ylabel('Actual Adoptions')
plt.show()

# def make_confusion_matrix(model, threshold=0.5):
#     # Predict class 1 if probability of being in class 1 is greater than threshold
#     # (model.predict(X_test) does this automatically with a threshold of 0.5)
#     y_predict = (model.predict_proba(X_test)[:, 1] >= threshold)
#     fraud_confusion = confusion_matrix(y_test, y_predict)
#     plt.figure(dpi=80)
#     sns.heatmap(fraud_confusion, cmap=plt.cm.Blues, annot=True, square=True, fmt='d',
#            xticklabels=['legit', 'fraud'],
#            yticklabels=['legit', 'fraud']);
#     plt.xlabel('prediction')
#     plt.ylabel('actual')


# Precision = TP / (TP + FP)
# Recall = TP/P = True positive rate
# false positive rate = FP / true negatives = FP / (FP + TN)

# ROC AUC
fpr, tpr, thresholds = roc_curve(label_test, logit.predict_proba(X2_test)[:, 1])

plt.plot(fpr, tpr,lw=2)
plt.plot([0,1],[0,1],c='violet',ls='--')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve for pet adoption')
plt.show()
print("ROC AUC score = ", roc_auc_score(label_test, logit.predict_proba(X2_test)[:, 1]))

# random forest fitting #######################

# test train split
X3_train, X3_test, label_train, label_test = train_test_split(train_df.drop(columns=['AdoptionSpeed', 'PetID']),
                                                            train_df['AdoptionSpeed'],
                                                            test_size=0.3, random_state=41)

rfmodel1 = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1)
rfmodel1.fit(X3_train, label_train)

print("The score for random forest is")
print("Training: {:.2f}%".format(100*f1_score(label_train, rfmodel1.predict(X3_train))))
print("Test set: {:.2f}%".format(100*f1_score(label_test, rfmodel1.predict(X3_test))))

# The score for random forest is
# Training: 52.91%
# Test set: 36.27%

# balanced class weights
rfmodel2 = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1,
                                  class_weight='balanced')
rfmodel2.fit(X3_train, label_train)

print("The score for random forest is")
print("Training: {:.2f}%".format(100*f1_score(label_train, rfmodel2.predict(X3_train))))
print("Test set: {:.2f}%".format(100*f1_score(label_test, rfmodel2.predict(X3_test))))

# The score for random forest is
# Training: 69.54%
# Test set: 56.55%

# plot first few levels of decision tree  ###############
# extract one single tree
# estimator = rfmodel2.estimators_[5]
# export_graphviz(estimator, out_file='tree.dot', max_depth=3,
#                 feature_names=train_df.drop(columns=['AdoptionSpeed', 'PetID']).columns,
#                 class_names=['adopted', 'not_adopted'],
#                 rounded=True, proportion=False,
#                 precision=2, filled=True)

# fit a single decision tree for illustration purposes
dt = DecisionTreeClassifier(min_samples_leaf=10, class_weight='balanced')
dt.fit(X3_train, label_train)
export_graphviz(dt, out_file='tree_dt.dot', max_depth=3,
                feature_names=train_df.drop(columns=['AdoptionSpeed', 'PetID']).columns,
                class_names=['adopted', 'not_adopted'],
                rounded=True, proportion=False,
                precision=2, filled=True)

# knn fitting ########################
knn_f1 = []
for k in range(3,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X3_train, label_train)
    knn_f1.append(tuple((k, f1_score(label_test, knn.predict(X3_test)))))
knn_f1

# [(3, 0.3842794759825327),
#  (4, 0.2714285714285714),
#  (5, 0.3657375934738274),
#  (6, 0.2611516626115166),
#  (7, 0.33764367816091956),
#  (8, 0.22591362126245845),
#  (9, 0.301659125188537),
#  (10, 0.2233502538071066),
#  (11, 0.2845973416731822),
#  (12, 0.21070811744386878),
#  (13, 0.26634382566585957),
#  (14, 0.21015761821366025),
#  (15, 0.260149130074565),
#  (16, 0.20598591549295775),
#  (17, 0.23659574468085104),
#  (18, 0.18996415770609318),
#  (19, 0.2302405498281787),
#  (20, 0.19099099099099096)]

# ordinal logistic ###############################

# training data
train_df = pd.read_csv('data/train.csv')

# drop a few columns and restrict to quantity == 1
train_df = train_df[train_df['Quantity']==1].drop(columns=['Description', 'VideoAmt', 'RescuerID', 'Name', 'Quantity'])
train_sent = pd.read_csv('sent_train.csv', index_col=0)

# keep only the records that have sentiment score
train_df = train_df.merge(train_sent, how='inner', left_on='PetID', right_index=True)

# one hot encode the dummy variables stored as values
train_df2 = pd.get_dummies(train_df, columns=['Type', 'Breed1', 'Breed2', 'Gender', 'Color1',
                                              'Color2', 'Color3', 'Vaccinated', 'Dewormed',
                                              'Sterilized', 'Health', 'State'])

X2_train, X2_test, label_train, label_test = train_test_split(train_df2.drop(columns=['AdoptionSpeed', 'PetID']),
                                                            train_df2['AdoptionSpeed'],
                                                            test_size=0.3, random_state=41)

std = StandardScaler()
std.fit(X2_train)
X2_train = std.transform(X2_train)
X2_test = std.transform(X2_test)

logit2 = LogisticRegression(
    multi_class='multinomial',
    solver='newton-cg',
    fit_intercept=True,
    C=0.95
).fit(X2_train, label_train)

print("The score for logistic regression is")
print("Training: {:.2f}%".format(100*f1_score(label_train, logit2.predict(X2_train), average='weighted')))
print("Test set: {:.2f}%".format(100*f1_score(label_test, logit2.predict(X2_test), average='weighted')))

cohen_kappa_score(label_test, logit2.predict(X2_test))