import os
import pandas as pd
import numpy as np

import scipy
import math

import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin, BaseEstimator 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelBinarizer, OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



# from sklearn.preprocessing import PolynomialFeatures

# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error

# from sklearn.ensemble import VotingClassifier

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC

# from sklearn.metrics import accuracy_score

# from sklearn.ensemble import BaggingClassifier
# from sklearn.tree import DecisionTreeClassifier

# from sklearn.ensemble import AdaBoostClassifier

# from sklearn.tree import DecisionTreeRegressor

# from sklearn.ensemble import GradientBoostingRegressor

# import sklearn.linear_model
# import scipy.optimize
# import sklearn.decomposition
# import sklearn.manifold
# import sklearn.model_selection


class CustomLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, X, y=0):
        self.encoder.fit(X)
        return self
    def transform(self, X, y=0):
        return self.encoder.transform(X)

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
split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, test_index in split.split(diabetic, diabetic['readmitted']):
    strat_train_set = diabetic.iloc[train_index]
    strat_test_set = diabetic.iloc[test_index]
    
diabetic_features = strat_train_set.drop("readmitted", axis=1)
diabetic_labels = strat_train_set["readmitted"].copy()

# Preprocessing pipeline
diabetic_num_to_cat_features = ['admission_type_id', 'discharge_disposition_id','admission_source_id']
diabetic_cat_to_num_features = ['max_glu_serum', 'A1Cresult']

diabetic_num_features = ['time_in_hospital', 'num_lab_procedures','num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

diabetic_cat_features = ['race', 'gender', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'metformin', 'repaglinide', 'nateglinide','chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide','tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol','troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone','metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed']

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

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(diabetic_num_to_cat_features + diabetic_cat_features + diabetic_cat_to_num_features)),
    ('imputer', SimpleImputer(strategy="constant")),
    ('encoder', OneHotEncoder(categories='auto'))

])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ('age_pipeline', age_pipeline),
    ("cat_pipeline", cat_pipeline),
])

diabetic_prepared = full_pipeline.fit_transform(diabetic_features)

