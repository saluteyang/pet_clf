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

plt.style.use('ggplot')
np.set_printoptions(suppress=True) # Suppress scientific notation where possible

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score,f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import cohen_kappa_score

# data pre-processing

# training data
train_df = pd.read_csv('data/train.csv')

# drop a few columns and restrict to quantity == 1
train_df = train_df[train_df['Quantity']==1].drop(columns=['Description', 'VideoAmt', 'RescuerID', 'Name', 'Quantity'])
train_sent = pd.read_csv('sent_train.csv', index_col=0)

# keep only the records that have sentiment score
train_df = train_df.merge(train_sent, how='inner', left_on='PetID', right_index=True)
train_df['AdoptionSpeed'] = train_df['AdoptionSpeed'].replace({1: 0, 2: 0, 3: 0, 4: 1})

####################
# one hot encode the dummy variables
train_df2 = pd.get_dummies(train_df, columns=['Type', 'Breed1', 'Breed2', 'Gender', 'Color1',
                                              'Color2', 'Color3', 'Vaccinated', 'Dewormed',
                                              'Sterilized', 'Health', 'State'])

X3_train, X3_test, label_train, label_test = train_test_split(train_df2.drop(columns=['AdoptionSpeed', 'PetID']),
                                                            train_df2['AdoptionSpeed'],
                                                            test_size=0.3, random_state=41)

std = StandardScaler()
std.fit(X3_train)
X3_train = std.transform(X3_train)
X3_test = std.transform(X3_test)

# error when changing scoring in logitCV to f1_score
logitCV = LogisticRegressionCV(cv=5, random_state=0).fit(X3_train, label_train)
logitCV.score(X3_train, label_train)   # 0.77 (acc)
logitCV.score(X3_test, label_test)   # 0.74 (acc)

f1_score(label_train, logitCV.predict(X3_train))   # 0.41
f1_score(label_test, logitCV.predict(X3_test))   # 0.36

numeric_cols = ['Age', 'MaturitySize', 'FurLength', 'Fee', 'PhotoAmt',
                'magnitude', 'score', 'AdoptionSpeed']

def plot_features(df):
    sample = (df[numeric_cols].sample(1000, random_state=44))
    sns.pairplot(sample, hue='AdoptionSpeed', plot_kws=dict(alpha=.3, edgecolor='none'))

plot_features(train_df)

feature_names = train_df2.drop(columns=['AdoptionSpeed', 'PetID']).columns
feature_coefs = pd.DataFrame({'feature': feature_names, 'coef': logitCV.coef_[0]})
feature_coefs['abs_coef'] = abs(feature_coefs['coef'])
feature_coefs.sort_values(by='abs_coef', ascending=False)

# how did the language score perform?
feature_coefs[feature_coefs['feature'].isin(['magnitude', 'score'])]


####################

# numeric dummies (baseline)
X_train, X_test, label_train, label_test = train_test_split(train_df.drop(columns=['AdoptionSpeed', 'PetID']),
                                                            train_df['AdoptionSpeed'],
                                                            test_size=0.3, random_state=41)

std = StandardScaler()
std.fit(X_train)
X_train = std.transform(X_train)
X_test = std.transform(X_test)

logit = LogisticRegression(C = 0.95)
logit.fit(X_train, label_train)
print("The score for logistic regression is")
print("Training: {:.2f}%".format(100*logit.score(X_train, label_train)))
print("Test set: {:.2f}%".format(100*logit.score(X_test, label_test)))

# The score for logistic regression is
# Training: 74.52%
# Test set: 73.04%

logitCV = LogisticRegressionCV(cv=5, random_state=0).fit(X_train, label_train)
logitCV.score(X_train, label_train)  # 0.74
logitCV.score(X_test, label_test)  # 0.73

# Print confusion matrix for logistic regression
logit_confusion = confusion_matrix(label_test, logit.predict(X_test))
plt.figure(dpi=150)
sns.heatmap(logit_confusion, cmap=plt.cm.Blues, annot=True, square=True,
           xticklabels=['adopted', 'not adopted'],
           yticklabels=['adopted', 'not adopted'])
plt.xlabel('Predicted Adoptions')
plt.ylabel('Actual Adoptions')
plt.show()

##########################

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

fpr, tpr, thresholds = roc_curve(label_test, logit.predict_proba(X_test)[:, 1])

plt.plot(fpr, tpr,lw=2)
plt.plot([0,1],[0,1],c='violet',ls='--')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve for pet adoption')
plt.show()
print("ROC AUC score = ", roc_auc_score(label_test, logit.predict_proba(X_test)[:, 1]))

##########################
# ordinal logistic

# training data
train_df = pd.read_csv('data/train.csv')

# drop a few columns and restrict to quantity == 1
train_df = train_df[train_df['Quantity']==1].drop(columns=['Description', 'VideoAmt', 'RescuerID', 'Name', 'Quantity'])
train_sent = pd.read_csv('sent_train.csv', index_col=0)

# keep only the records that have sentiment score
train_df = train_df.merge(train_sent, how='inner', left_on='PetID', right_index=True)


X2_train, X2_test, label_train, label_test = train_test_split(train_df.drop(columns=['AdoptionSpeed', 'PetID']),
                                                            train_df['AdoptionSpeed'],
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
)
logit2.fit(X2_train, label_train)

logit2.score(X2_train, label_train)
logit2.score(X2_test, label_test)

logit2.predict_proba(X2_test)[5:]
logit2.predict(X2_test)[5:]

cohen_kappa_score(label_test, logit2.predict(X_test))