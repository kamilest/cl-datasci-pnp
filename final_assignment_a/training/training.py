import os
import pandas as pd
import numpy as np

import scipy
import math

import sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin, BaseEstimator 

from sklearn.dummy import DummyClassifier

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

np.random.seed(42)

diabetic = pd.read_csv('diabetes/diabetic_data_balanced.csv')

# Anonymise
diabetic = diabetic.groupby('patient_nbr', group_keys=False) \
                    .apply(lambda df: df.sample(1))


# Train test split
X, y = diabetic.drop('readmitted', axis=1), diabetic['readmitted']

split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, test_index in split.split(X, y):
     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    

# PREPROCESSING PIPELINE
diabetic_num_to_cat_features = ['admission_type_id', 'discharge_disposition_id','admission_source_id']

diabetic_cat_to_num_features = ['max_glu_serum', 'A1Cresult']

diabetic_num_features = ['time_in_hospital', 'num_lab_procedures','num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']


# CHANGE DIABETIC DRUGS TO EITHER FULL AND EMPTY ARRAY
# FOR FULL AND REDUCED FEATURE SETS

# diabetic_drugs = ['medical_specialty', 'metformin', 'repaglinide', 'nateglinide','chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide','tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol','troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone','metformin-rosiglitazone', 'metformin-pioglitazone']

diabetic_drugs = []

diabetic_cat_features = ['race', 'gender', 'change', 'diabetesMed']
diabetic_diag_features = ['diag_1', 'diag_2', 'diag_3']

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(diabetic_num_features)),
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

age_pipeline = Pipeline([
    ('selector', DataFrameSelector(['age'])),
    ('ordinal_encoder', OrdinalEncoder(categories=[['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']])),
    ('std_scaler', StandardScaler()),
])

class DiagEncoder(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = OneHotEncoder(*args, **kwargs, categories='auto', sparse=False)
    def fit(self, X, y=0):
        X = [[self.get_type(icd_str) for icd_str in x] for x in X]
        self.encoder.fit(X)
        return self
    def transform(self, X, y=0):
        X = [[self.get_type(icd_str) for icd_str in x] for x in X]
        return self.encoder.transform(X)
    def get_type(self, icd_str):
        if isinstance(icd_str, float) and math.isnan(icd_str):
            return('missing')
        elif icd_str.isnumeric():
            icd = int(icd_str)
        elif icd_str[:3].isnumeric():
            icd = int(icd_str[:3])
        else:
            return 'other'

        if (icd >= 390 and icd <= 459 or icd == 785):
            return 'circulatory'
        elif (icd >= 520 and icd <= 579 or icd == 787):
            return 'digestive'
        elif (icd >= 580 and icd <= 629 or icd == 788):
            return 'genitourinary'
        elif (icd == 250):
            return 'diabetes'
        elif (icd >= 800 and icd <= 999):
            return 'injury'
        elif (icd >= 710 and icd <= 739):
            return 'musculoskeletal'
        elif (icd >= 140 and icd <= 239):
            return 'neoplasms'
        elif (icd >= 460 and icd <= 519 or icd == 786):
            return 'respiratory'
        else:
            return 'other'
        

diag_pipeline = Pipeline([
    ('selector', DataFrameSelector(diabetic_diag_features)),
    ('diag_encoder', DiagEncoder()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(diabetic_num_to_cat_features + diabetic_cat_features + diabetic_drugs + diabetic_cat_to_num_features)),
    ('imputer', SimpleImputer(strategy='constant')),
    ('encoder', OneHotEncoder(categories='auto', sparse=False))

])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ('age_pipeline', age_pipeline),
    ('diag_pipeline', diag_pipeline),
    ("cat_pipeline", cat_pipeline),
])

X_train = full_pipeline.fit_transform(X_train)


# MACHINE LEARNING ALGORITHMS
print(X_train.shape)


# MULTI-CLASS CLASSIFIERS 

sgd = SGDClassifier(random_state=42)
cv_sgd = cross_val_score(sgd, X_train, y_train, cv=5, scoring='accuracy')
# print(cv_sgd, np.mean(cv_sgd))
print('cv_sgd', np.mean(cv_sgd))

log_reg = LogisticRegression(random_state=42, multi_class='ovr', solver='liblinear')
cv_log_reg = cross_val_score(log_reg, X_train, y_train, cv=5, scoring='accuracy')
# print(cv_log_reg, np.mean(cv_log_reg))
print('cv_log_reg', np.mean(cv_log_reg))


gnb = GaussianNB()
cv_gnb = cross_val_score(gnb, X_train, y_train, cv=5, scoring='accuracy')
# print(cv_gnb, np.mean(cv_gnb))
print('cv_gnb', np.mean(cv_gnb))

baseline = DummyClassifier(random_state=42)
cv_baseline = cross_val_score(baseline, X_train, y_train, cv=5, scoring='accuracy')
# print(cv_baseline, np.mean(cv_baseline))
print("cv_baseline", np.mean(cv_baseline))

# KERNEL TRICK 

from sklearn.kernel_approximation import RBFSampler
rbf_features = RBFSampler(gamma=1, n_components=100, random_state=42)
X_train_features = rbf_features.fit_transform(X_train)
print(X_train_features.shape)

sgd = SGDClassifier(random_state=42)
cv_sgd = cross_val_score(sgd, X_train_features, y_train, cv=5, scoring='accuracy')
print('cv_sgd', np.mean(cv_sgd))

log_reg = LogisticRegression(random_state=42, multi_class='ovr', solver='liblinear')
cv_log_reg = cross_val_score(log_reg, X_train_features, y_train, cv=5, scoring='accuracy')
print('cv_log_reg', np.mean(cv_log_reg))

gnb = GaussianNB()
cv_gnb = cross_val_score(gnb, X_train_features, y_train, cv=5, scoring='accuracy')
print('cv_gnb', np.mean(cv_gnb))


# GRID SEARCH LOGISTIC REGRESSION
# specify the range of hyperparameter values for the grid search to try out 

param_grid = {'penalty': ['l1', 'l2'], 'C': [0.25, 0.5, 1.0]}

forest_reg = LogisticRegression(random_state=42, solver='liblinear', multi_class='ovr')
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                          scoring="accuracy")
grid_search.fit(X_train, y_train)

print(grid_search.best_params_, grid_search.best_score_)


# VOTING CLASSIFIER

log_clf = LogisticRegression(random_state=42, multi_class='ovr', solver='liblinear')
rnd_clf = RandomForestClassifier(random_state=42, n_estimators=100)
svm_clf = SVC(random_state=42, gamma='scale')

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')

cv_voting = cross_val_score(voting_clf, X_train, y_train, cv=5, scoring="accuracy")
print('cv_voting', np.mean(cv_voting))


# BAGGING AND PASTING ENSEMBLES

# Random Forest Classifier 
# (BaggingClassifier with DecisionTreeClassifier as base)

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, 
                                 n_jobs=-1, random_state=42, oob_score=True)
rnd_clf.fit(X_train, y_train)
cv_rnd_clf = cross_val_score(rnd_clf, X_train, y_train, cv=5, scoring="accuracy")
print('cv_rnd_clf', np.mean(cv_rnd_clf), '(oob score {})'.format(rnd_clf.oob_score_))


rnd_clf_pasting = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, 
                                 n_jobs=-1, random_state=42, bootstrap=False)
rnd_clf_pasting.fit(X_train, y_train)
cv_rnd_clf_pasting = cross_val_score(rnd_clf_pasting, X_train, y_train, cv=5, scoring="accuracy")
print('cv_rnd_clf_pasting', np.mean(cv_rnd_clf_pasting))


# ADABOOST CLASSIFIER

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
cv_ada_clf = cross_val_score(ada_clf, X_train, y_train, cv=5, scoring="accuracy")
print('cv_ada_clf', np.mean(cv_ada_clf))

# GRADIENT BOOSTING

gb_clf = GradientBoostingClassifier(max_depth=10, n_estimators=100, learning_rate=0.1, random_state=42)
cv_gb_clf = cross_val_score(gb_clf, X_train, y_train, cv=5, scoring="accuracy")
print('cv_gb_clf', np.mean(cv_gb_clf))


# GRADIENT BOOSTING WITH EARLY STOPPING

# early stopping
gbes_clf = GradientBoostingClassifier(max_depth=10, validation_fraction=0.1, n_iter_no_change=10, tol=0.01, n_estimators=100, learning_rate=0.1, random_state=42)
cv_gbes_clf = cross_val_score(gbes_clf, X_train, y_train, cv=5, scoring="accuracy")
print('cv_gbes_clf', np.mean(cv_gbes_clf))

# AUTOML PIPELINE FOR HYPERPARAMETER TUNING

tpot_best_clf = GradientBoostingClassifier(learning_rate=0.01, max_depth=10, max_features=0.2, min_samples_leaf=1, min_samples_split=14, n_estimators=100, subsample=0.9000000000000001, random_state=42)

cv_tpot_best_clf = cross_val_score(tpot_best_clf, X_train, y_train, cv=5, scoring="accuracy")
print('cv_tpot_best_clf', np.mean(cv_tpot_best_clf))



