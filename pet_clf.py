import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
np.set_printoptions(suppress=True)  # Suppress scientific notation where possible

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.naive_bayes import BernoulliNB

from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import keras
from keras import backend as K

# data pre-processing #########################
train_df = pd.read_csv('data/train.csv')
columns_to_drop = ['Description', 'VideoAmt', 'RescuerID', 'Name', 'Quantity']
train_df = train_df[train_df['Quantity'] == 1].drop(columns=columns_to_drop)
train_sent = pd.read_csv('sent_train.csv', index_col=0)

# keep only the records that have sentiment score
train_df = train_df.merge(train_sent, how='inner', left_on='PetID', right_index=True)
train_df['AdoptionSpeed'] = train_df['AdoptionSpeed'].replace({1: 0, 2: 0, 3: 0, 4: 1})

# one hot encode the dummy variables stored as values
onehot_columns = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
                  'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State']
train_df2 = pd.get_dummies(train_df, columns=onehot_columns)

# test train split
X_train, X_test, label_train, label_test = train_test_split(train_df2.drop(columns=['AdoptionSpeed', 'PetID']),
                                                            train_df2['AdoptionSpeed'],
                                                            test_size=0.3, random_state=41)

# standard scaler (omit for random forest)
std = StandardScaler()
std.fit(X_train)
X_train = std.transform(X_train)
X_test = std.transform(X_test)

# logistic model baseline ###########################
# logistic with custom regularization
logit = LogisticRegression(C=0.95)
logit.fit(X_train, label_train)
print("The score for logistic regression is")
print("Training: {:.2f}%".format(100*f1_score(label_train, logit.predict(X_train))))
print("Test set: {:.2f}%".format(100*f1_score(label_test, logit.predict(X_test))))

# The score for logistic regression is
# Training: 41.18%
# Test set: 35.47%

# logistic with balanced class weights
logit = LogisticRegression(C=0.95, class_weight='balanced')
logit.fit(X_train, label_train)
print("The score for logistic regression is")
print("Training: {:.2f}%".format(100*f1_score(label_train, logit.predict(X_train))))
print("Test set: {:.2f}%".format(100*f1_score(label_test, logit.predict(X_test))))

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
grouped_by_species = grouped_by_species.merge(breed_labels[['BreedID', 'BreedName']],
                                              left_on='breed', right_on='BreedID')

plt.bar(grouped_by_species[grouped_by_species['Species']=='cat']['BreedName'],
        grouped_by_species[grouped_by_species['Species']=='cat']['coef'])
plt.xticks(rotation=90)
plt.show()

plt.bar(grouped_by_species[grouped_by_species['Species']=='dog']['BreedName'],
        grouped_by_species[grouped_by_species['Species']=='dog']['coef'])
plt.xticks(rotation=90)
plt.show()

# tree models #############################################
# excluding cats from here on
# add breed groups
import unicodedata
breed_groups = pd.read_csv('data/breed_groups.csv', encoding='latin_1')
for i in range(len(breed_groups['BreedName'])):
    breed_groups['BreedName'][i] = unicodedata.normalize('NFKD', breed_groups['BreedName'][i]).\
        encode('ascii', 'ignore').decode('utf-8')

train_df_wg = train_df.merge(breed_labels[['BreedID', 'BreedName']], how='left', left_on='Breed1', right_on='BreedID').\
    merge(breed_groups, how='left', on='BreedName')

# how many dog breeds were a match to groups?
# train_df_wg[(train_df_wg['Type'] == 1) & (train_df_wg['BreedName'] != 'Mixed Breed')]['BreedID'].count()
# train_df_wg[(train_df_wg['Type'] == 1)]['BreedGroup'].dropna().count()

# Mixed Breed retain the name in BreedGroup feature; 'Unknown', if no mapping from BreedName to BreedGroup
train_df_sml = train_df_wg[(train_df_wg['Type'] == 1)]
train_df_sml = train_df_sml[train_df_sml['BreedName'].notna()]
train_df_sml.loc[train_df_sml['BreedName'] == 'Mixed Breed', 'BreedGroup'] = 'Mixed Breed'
train_df_sml['BreedGroup'] = ['Unknown' if x is np.nan else x for x in train_df_sml['BreedGroup'].tolist()]
train_df_sml['BreedGroup'] = [x.strip() for x in train_df_sml['BreedGroup']]

train_df_sml = train_df_sml.drop(columns=['Type', 'Breed1', 'Breed2', 'PetID', 'BreedID'])
train_df_sml = pd.get_dummies(train_df_sml, columns=['BreedName', 'BreedGroup', 'Gender', 'Color1',
                                                     'Color2', 'Color3', 'Vaccinated', 'Dewormed',
                                                     'Sterilized', 'Health', 'State'])

# test train split
Xs_train, Xs_test, label_train_s, label_test_s = train_test_split(train_df_sml.drop(columns=['AdoptionSpeed']),
                                                                  train_df_sml['AdoptionSpeed'],
                                                                  test_size=0.3, random_state=41)

# balanced class weights Random Forest
rfmodel_fin = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1,
                                     class_weight='balanced')
rfmodel_fin.fit(Xs_train, label_train_s)

print("The score for random forest is")
print("Training: {:.2f}%".format(100*f1_score(label_train_s, rfmodel_fin.predict(Xs_train))))
print("Test set: {:.2f}%".format(100*f1_score(label_test_s, rfmodel_fin.predict(Xs_test))))


# remove BreedName to see impact of absence  #####################
breedname_cols = [col for col in Xs_train.columns if col[:10] == 'BreedName_']
Xs_train, Xs_test = Xs_train.drop(columns=breedname_cols), Xs_test.drop(columns=breedname_cols)
rfmodel_fin = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1,
                                     class_weight='balanced')
rfmodel_fin.fit(Xs_train, label_train_s)

print("The score for random forest is")
print("Training: {:.2f}%".format(100*f1_score(label_train_s, rfmodel_fin.predict(Xs_train))))
print("Test set: {:.2f}%".format(100*f1_score(label_test_s, rfmodel_fin.predict(Xs_test))))

# the scores are similar with either BreedName or BreedGroup
# BreedGroups are Companion, Guardian, Gun, Herding, Northern, Scenthound, Sighthound, Terrier

# feature importance plot (unknown breed dogs only, dummified categories)
feature_names = train_df_sml.drop(columns=['AdoptionSpeed']).columns
importances = rfmodel_fin.feature_importances_
indices = np.argsort(importances)

features_plot = 6
plt.figure()
plt.barh(range(len(indices))[-features_plot:], importances[indices][-features_plot:], color='b', align='center')
plt.yticks(range(len(indices))[-features_plot:], [feature_names[i] for i in indices][-features_plot:])
plt.xlabel('Relative Importance')
plt.title('Feature Importances')
plt.savefig('feature_import.png', dpi=600, bbox_inches="tight")
plt.show()

# how did the language score perform?
feature_coefs[feature_coefs['feature'].isin(['magnitude', 'score'])]

# Print confusion matrix for regression
confusion = confusion_matrix(label_test_s, rfmodel_fin.predict(Xs_test))
plt.figure(dpi=150)
sns.heatmap(confusion, cmap=plt.cm.Blues, annot=True, square=True, fmt='g',
            xticklabels=['adopted', 'not adopted'],
            yticklabels=['adopted', 'not adopted'])
plt.xlabel('Predicted Adoptions')
plt.ylabel('Actual Adoptions')
plt.show()

# Precision = TP / (TP + FP)
# Recall = TP/P = True positive rate
# false positive rate = FP / true negatives = FP / (FP + TN)

# ROC AUC
fpr, tpr, thresholds = roc_curve(label_test_s, rfmodel_fin.predict_proba(Xs_test)[:, 1])

plt.plot(fpr, tpr, lw=2)
plt.plot([0, 1], [0, 1], c='violet', ls='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve for pet adoption')
plt.show()
print("ROC AUC score = ", roc_auc_score(label_test_s, rfmodel_fin.predict_proba(Xs_test)[:, 1]))

# tree-based model selection ##################################
# try a few other tree models as well as stacking different models
et = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1,
                          class_weight='balanced')
adb = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'),
                         n_estimators=100)

models = ['rfmodel_fin', 'et', 'adb']
for model_name in models:
    curr_model = eval(model_name)
    curr_model.fit(Xs_train, label_train_s)
    print(f'{model_name} score: {f1_score(label_test_s, curr_model.predict(Xs_test))}')

# create meta-classifier
stacked = StackingClassifier(classifiers=[eval(n) for n in models],
                             meta_classifier=BernoulliNB(),
                             use_probas=False)
stacked.fit(Xs_train, label_train_s)

print(f'stacked score: {f1_score(label_test_s, stacked.predict(Xs_test))}')

# xgboost ######################################
# random oversampling of under-represented class
ros = RandomOverSampler(random_state=0)
X_ros, y_ros = ros.fit_sample(Xs_train, label_train_s)

# ROS removed column labels, put them back
X_ros = pd.DataFrame(X_ros)
X_ros.columns = Xs_train.columns

X_ros_train, X_ros_val, label_ros_train, label_ros_val = train_test_split(X_ros, y_ros, test_size=0.2, random_state=41)

# xgbc = xgb.XGBClassifier(n_estimators=100,
#                          max_depth=5,
#                          objective='binary:logistic',
#                          learning_rate=0.05,
#                          subsample=0.8
#                          # min_child_weight=3
#                          # colsample_bytree=0.8
#                          )

xgbc = xgb.XGBClassifier(n_estimators=200,
                         max_depth=6,
                         objective='binary:logistic',
                         learning_rate=0.3,
                         subsample=0.8,
                         min_child_weight=1,
                         colsample_bytree=1,
                         gamma=0.5
                         )

eval_set = [(X_ros_train, label_ros_train), (X_ros_val, label_ros_val)]
xgbc.fit(X_ros_train, label_ros_train, eval_set=eval_set, eval_metric='aucpr')

f1_score(label_test_s, xgbc.predict(Xs_test))

# grid search for xgboost
from sklearn.model_selection import GridSearchCV

fin_model = xgb.XGBClassifier(objective='binary:logistic')
params = {
    'n_estimators': [100, 200],
    'min_child_weight': [1, 3],
    'gamma': [0.5, 1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.05, 0.1, 0.2, 0.3]
        }
grid_search = GridSearchCV(fin_model, params, scoring="f1", n_jobs=-1, cv=5)
grid_result = grid_search.fit(X_ros, y_ros)

# summarize xgboost grid search results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f"{mean} ({stdev}) with: {param}")

# Best: 0.889289 using {'colsample_bytree': 1.0, 'gamma': 0.5, 'learning_rate': 0.3,
# 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 200, 'subsample': 0.8}

# fit a single decision tree for illustration purposes
dt = DecisionTreeClassifier(min_samples_leaf=10, class_weight='balanced')
dt.fit(X_train, label_train)
export_graphviz(dt, out_file='tree_dt.dot', max_depth=3,
                feature_names=train_df.drop(columns=['AdoptionSpeed', 'PetID']).columns,
                class_names=['adopted', 'not_adopted'],
                rounded=True, proportion=False,
                precision=2, filled=True)

# neural network classifier  #####################
# random over sampling of under-represented class
ros = RandomOverSampler(random_state=0)
X_ros, y_ros = ros.fit_sample(Xs_train, label_train_s)

X_ros_train, X_ros_val, label_ros_train, label_ros_val = train_test_split(X_ros, y_ros, test_size=0.2, random_state=41)

# setting up the model and metric
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
    nn_model.add(keras.layers.Dense(units=20, input_dim=X_ros_train.shape[1], activation='tanh'))
    nn_model.add(keras.layers.Dense(units=20, activation='tanh'))
    nn_model.add(keras.layers.Dense(units=1, activation='sigmoid'))

    nn_model.compile(loss='binary_crossentropy',
                     optimizer= "adam",
                     metrics=[f1])
    return nn_model

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
h = nn_model.fit(X_ros_train, label_ros_train, batch_size=100, epochs=200,
                 validation_data=(X_ros_val, label_ros_val))
nn_model.evaluate(Xs_test, label_test_s)

nn_model = build_model()
h2 = nn_model.fit(X_ros_train, label_ros_train, batch_size=100, epochs=125)
nn_model.evaluate(Xs_test, label_test_s)

nn_model = build_model()
h3 = nn_model.fit(X_ros_train, label_ros_train, batch_size=20, epochs=100,
                  validation_data=(X_ros_val, label_ros_val))
nn_model.evaluate(Xs_test, label_test_s)

# nn_model.metrics_names
# ['loss', 'f1']
# [0.8370354568012559, 0.5207267891824576]

plot_train_val_scores(h3)
plt.show()
