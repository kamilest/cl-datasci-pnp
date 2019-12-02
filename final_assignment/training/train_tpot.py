import os
import pandas as pd
import numpy as np

import scipy
import math

import sklearn
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin, BaseEstimator 

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

from tpot import TPOTClassifier



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
    
# diabetic_features = strat_train_set.drop("readmitted", axis=1)
# diabetic_labels = strat_train_set["readmitted"].copy()

# PREPROCESSING PIPELINE
diabetic_num_to_cat_features = ['admission_type_id', 'discharge_disposition_id','admission_source_id']

diabetic_cat_to_num_features = ['max_glu_serum', 'A1Cresult']

diabetic_num_features = ['time_in_hospital', 'num_lab_procedures','num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

# diabetic_num_features = ['time_in_hospital', 'num_lab_procedures','num_procedures', 'num_medications', 'number_diagnoses']

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

tpot_clf = TPOTClassifier(generations=5, population_size=20, cv=5, random_state=42, verbosity=2)

tpot_clf.fit(X_train, y_train)
print(tpot_clf.score(X_train, y_train))

cv_tpot_clf = cross_val_score(tpot_clf, X_train, y_train, cv=5, scoring="accuracy")
print('cv_tpot_clf', np.mean(cv_tpot_clf))