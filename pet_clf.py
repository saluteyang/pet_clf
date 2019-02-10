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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
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

# The score for logistic regression is
# Training: 41.18%
# Test set: 35.47%

# logistic with balanced class weights
logit = LogisticRegression(C=0.95, class_weight='balanced')
logit.fit(X2_train, label_train)
print("The score for logistic regression is")
print("Training: {:.2f}%".format(100*f1_score(label_train, logit.predict(X2_train))))
print("Test set: {:.2f}%".format(100*f1_score(label_test, logit.predict(X2_test))))

# The score for logistic regression is
# Training: 53.68%
# Test set: 51.40%

# interpretation and diagnostics ###################
# select numerical columns for pairplot
numeric_cols = ['Age', 'MaturitySize', 'FurLength', 'Fee', 'PhotoAmt',
                'magnitude', 'score', 'AdoptionSpeed']

numeric_cols = ['Age', 'PhotoAmt', 'AdoptionSpeed']

def plot_features(df):
    sample = df[numeric_cols].sample(1000, random_state=44)
    sns.pairplot(sample, hue='AdoptionSpeed', plot_kws=dict(alpha=.3, edgecolor='none'))

plot_features(train_df)
plt.savefig('age_photoamt.png', dpi=600, bbox_inches="tight")
plt.show()

# show coefs for logistic regression with feature names
feature_names = train_df2.drop(columns=['AdoptionSpeed', 'PetID']).columns
feature_coefs = pd.DataFrame({'feature': feature_names, 'coef': logit.coef_[0]})
feature_coefs['abs_coef'] = abs(feature_coefs['coef'])
feature_coefs.sort_values(by='abs_coef', ascending=False, inplace=True)

# breed subset features
breed_labels = pd.read_csv('data/breed_labels.csv')

breed_coefs = feature_coefs[(feature_coefs['feature'].str.contains('Breed1')) & (feature_coefs['abs_coef'] > 0.1)]
breed_coefs[['feature', 'breed']] = breed_coefs['feature'].str.split('_', expand=True)
breed_coefs['Species'] = ['cat' if (int(x) > 240) & (int(x) < 307) else 'dog' for x in breed_coefs['breed']]
grouped_by_species = breed_coefs.groupby(['Species'])['breed', 'coef', 'abs_coef']
grouped_by_species = grouped_by_species.apply(lambda _df: _df.sort_values(by=['coef'])).reset_index()


grouped_by_species['breed'] = grouped_by_species['breed'].astype('int')
grouped_by_species = grouped_by_species.merge(breed_labels[['BreedID', 'BreedName']], left_on='breed', right_on='BreedID')

plt.bar(grouped_by_species[grouped_by_species['Species']=='cat']['BreedName'],
        grouped_by_species[grouped_by_species['Species']=='cat']['coef'])
plt.xticks(rotation=90)
plt.show()

plt.bar(grouped_by_species[grouped_by_species['Species']=='dog']['BreedName'],
        grouped_by_species[grouped_by_species['Species']=='dog']['coef'])
plt.xticks(rotation=90)
plt.show()

# add breed groups
import unicodedata
breed_groups = pd.read_csv('data/breed_groups.csv', encoding='latin_1')
for i in range(len(breed_groups['BreedName'])):
    breed_groups['BreedName'][i] = unicodedata.normalize('NFKD', breed_groups['BreedName'][i]).encode('ascii', 'ignore').decode('utf-8')

train_df_wg = train_df.merge(breed_labels[['BreedID', 'BreedName']], how='left', left_on='Breed1', right_on='BreedID').\
    merge(breed_groups, how='left', on='BreedName')

# how many dog breeds were a match to groups?
train_df_wg[(train_df_wg['Type']==1) & (train_df_wg['BreedName']!='Mixed Breed')]['BreedID'].count()
train_df_wg[(train_df_wg['Type']==1)]['BreedGroup'].dropna().count()

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

# age count
train_df.groupby(['Age'])[['Breed1']].count().sort_values(by='Breed1', ascending=False)

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

# balanced class weights (dummified)
rfmodel3 = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1,
                                  class_weight='balanced')
rfmodel3.fit(X2_train, label_train)

print("The score for random forest is")
print("Training: {:.2f}%".format(100*f1_score(label_train, rfmodel3.predict(X2_train))))
print("Test set: {:.2f}%".format(100*f1_score(label_test, rfmodel3.predict(X2_test))))

# The score for random forest is
# Training: 58.06%
# Test set: 53.29%

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

rfmodel4 = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1,
                                  class_weight='balanced')
rfmodel4.fit(Xf_train, label_train)

print("The score for random forest is")
print("Training: {:.2f}%".format(100*f1_score(label_train, rfmodel4.predict(Xf_train))))
print("Test set: {:.2f}%".format(100*f1_score(label_test, rfmodel4.predict(Xf_test))))

# The score for random forest is
# Training: 70.00%
# Test set: 56.18%

# balanced class weights (with age z score)
z_score = lambda x: (x - x.mean())/x.std()
train_df_trans = train_df.copy()
train_df_trans['Age'] = train_df_trans.groupby(['Type'])['Age'].transform(z_score)

# test train split
Xt_train, Xt_test, label_train, label_test = train_test_split(train_df_trans.drop(columns=['AdoptionSpeed', 'PetID']),
                                                            train_df_trans['AdoptionSpeed'],
                                                            test_size=0.3, random_state=41)

rfmodel5 = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1,
                                  class_weight='balanced')
rfmodel5.fit(Xt_train, label_train)

print("The score for random forest is")
print("Training: {:.2f}%".format(100*f1_score(label_train, rfmodel5.predict(Xt_train))))
print("Test set: {:.2f}%".format(100*f1_score(label_test, rfmodel5.predict(Xt_test))))

# The score for random forest is
# Training: 69.55%
# Test set: 55.48%

et = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1,
                            class_weight='balanced')
et.fit(X3_train, label_train)
print("The score for extra trees is")
print("Training: {:.2f}%".format(100*f1_score(label_train, et.predict(X3_train))))
print("Test set: {:.2f}%".format(100*f1_score(label_test, et.predict(X3_test))))

# The score for extra trees is
# Training: 65.25%
# Test set: 54.28%

adb = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'),
                         n_estimators=100)
adb.fit(X3_train, label_train)
print("The score for AdaBoost is")
print("Training: {:.2f}%".format(100*f1_score(label_train, adb.predict(X3_train))))
print("Test set: {:.2f}%".format(100*f1_score(label_test, adb.predict(X3_test))))

# The score for AdaBoost is
# Training: 68.89%
# Test set: 53.83%

# no option to adjust class weights for gradient boosted trees
# gbt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5,
#                                  max_depth=3)
# gbt.fit(X3_train, label_train)
# print("The score for GB is")
# print("Training: {:.2f}%".format(100*f1_score(label_train, gbt.predict(X3_train))))
# print("Test set: {:.2f}%".format(100*f1_score(label_test, gbt.predict(X3_test))))

# The score for GB is
# Training: 61.72%
# Test set: 45.64%

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

# neural network classifier  #####################
import os
import keras
from keras import backend as K
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# random over sampling of under-represented class
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X2_train, label_train)

# carve out a validation set to tune number of epochs
X_resampled_train, X_resampled_val, label_resampled_train, label_resampled_val = train_test_split(X_resampled,
                                                                                                   y_resampled,
                                                                                                   test_size=0.2,
                                                                                                   random_state=41)

# setting up the model ########################

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def build_model():
    nn_model = keras.models.Sequential()
    nn_model.add(keras.layers.Dense(units=20, input_dim=X_resampled_train.shape[1], activation='tanh'))
    nn_model.add(keras.layers.Dense(units=20, activation='tanh'))
    nn_model.add(keras.layers.Dense(units=1, activation='sigmoid'))

    nn_model.compile(loss='binary_crossentropy',
              optimizer= "adam",
              metrics=[f1])
    return nn_model

nn_model = build_model()
h = nn_model.fit(X_resampled_train, label_resampled_train, batch_size=100, epochs=200,
                 validation_data=(X_resampled_val, label_resampled_val))
nn_model.evaluate(X2_test, label_test)
# [1.1126724123954772, 0.44647203882535297]
# overfitting: train f1 in the last iter is 0.84

# create plot of loss/metric over epochs for training and validation
def plot_train_val_scores(model_h):
    score = model_h.history['f1']
    val_score = model_h.history['val_f1']
    history_dict = model_h.history
    epochs = range(1, len(history_dict['f1']) + 1)

    plt.plot(epochs, score, 'bo', label='training f1 score')  # blue dot
    plt.plot(epochs, val_score, 'b', label='validation f1 score')
    plt.title('training and validation f1 score')
    plt.xlabel('Epochs')
    plt.ylabel('f1')
    plt.legend()

nn_model = build_model()
h2 = nn_model.fit(X_resampled_train, label_resampled_train, batch_size=100, epochs=125)
nn_model.evaluate(X2_test, label_test)
# [1.2461713705744062, 0.4506803986572084]

nn_model = build_model()
h3 = nn_model.fit(X_resampled_train, label_resampled_train, batch_size=20, epochs=100,
                  validation_data=(X_resampled_val, label_resampled_val))
nn_model.evaluate(X2_test, label_test)
# [1.4074761067117965, 0.4550816105235191]

plot_train_val_scores(h3)
plt.show()