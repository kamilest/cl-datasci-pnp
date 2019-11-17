import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import TransformerMixin, BaseEstimator 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(42)

def load_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_data("/Users/kamilestankeviciute/Google Drive/Part II/_MT19/Advanced Data Science/cl-datasci-pnp-private/DSPNP_practical1/housing/")

rooms_id, bedrooms_id, population_id, household_id = 3, 4, 5, 6

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace = True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


class CustomLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, X, y=0):
        self.encoder.fit(X)
        return self
    def transform(self, X, y=0):
        return self.encoder.transform(X)

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_rooms = True): # note no *args and **kwargs used this time
        self.add_bedrooms_per_rooms = add_bedrooms_per_rooms
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_id] / X[:, household_id]
        bedrooms_per_household = X[:, bedrooms_id] / X[:, household_id]
        population_per_household = X[:, population_id] / X[:, household_id]
        if self.add_bedrooms_per_rooms:
            bedrooms_per_rooms = X[:, bedrooms_id] / X[:, rooms_id]
            return np.c_[X, rooms_per_household, bedrooms_per_household, 
                         population_per_household, bedrooms_per_rooms]
        else:
            return np.c_[X, rooms_per_household, bedrooms_per_household, 
                         population_per_household]

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

def drop_uncorrelated_features(housing, housing_labels, threshold=0.1, inplace=False):
    corr_matrix = housing.corrwith(housing_labels)
    result = housing.copy()
    for feature_name in corr_matrix.keys():
        if abs(corr_matrix[feature_name]) <= threshold:
            result = result.drop(feature_name, axis=1, inplace=inplace)
    return result


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# housing = drop_uncorrelated_features(housing, housing_labels)

housing_num = housing.drop("ocean_proximity", axis=1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', CustomLabelBinarizer()),
    ])


full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)

metrics = {}

# LINEAR REGRESSION
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)

metrics["lin_rmse"] = lin_rmse
print("lin_mrse on training set: {}".format(lin_rmse))


# POLYNOMIAL FEATURES 
model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                  ('linear', LinearRegression())])

model = model.fit(housing_prepared, housing_labels)
housing_predictions = model.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)

metrics["poly_rmse"] = lin_rmse
print("lin_mrse on training set with polynomial features {}".format(lin_rmse))


# DECISION TREE
tree_reg = DecisionTreeRegressor()
tree_reg = tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_mse = np.sqrt(tree_mse)

print("mrse on training set with decision tree regressor {}".format(tree_mse))


# RANDOM FOREST
forest_reg = RandomForestRegressor(n_estimators=10)
    
# CROSS VALIDATION
def analyse_cv(model):   
    scores = cross_val_score(model, housing_prepared, housing_labels,
                             scoring = "neg_mean_squared_error", cv=10)

    # cross-validation expects utility function (greater is better)
    # rather than cost function (lower is better), so the scores returned
    # are negative as they are the opposite of MSE
    sqrt_scores = np.sqrt(-scores) 
    # print("Scores:", sqrt_scores)
    # print("Mean:", sqrt_scores.mean())
    # print("Standard deviation:", sqrt_scores.std())
    return sqrt_scores.mean()
    
metrics["tree_cv"] = analyse_cv(tree_reg)
metrics["lin_cv"] = analyse_cv(lin_reg)
metrics["forest_cv"] = analyse_cv(forest_reg)
print("cv mean on decision tree {}".format(metrics["tree_cv"]))
print("cv mean on linear regression {}".format(metrics["lin_cv"]))
print("cv mean on random forest {}".format(metrics["forest_cv"]))


# GRID SEARCH ON RANDOM FOREST

# specify the range of hyperparameter values for the grid search to try out 
param_grid = {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                          scoring="neg_mean_squared_error")
grid_search.fit(housing_prepared, housing_labels)
print("grid search best params {}".format(grid_search.best_params_))

cv_results = grid_search.cv_results_
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

extra_attribs = ['rooms_per_household', 'bedrooms_per_household', 'population_per_household', 'bedrooms_per_rooms']
cat_one_hot_attribs = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

final_model = grid_search.best_estimator_

# final_model = lin_reg
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

metrics["final_rmse"] = final_rmse
print("rmse on random forest grid search {}".format(final_rmse))
print(metrics)

# print("lin_reg: intercept_ = {} \n coef_ = {}".format(lin_reg.intercept_, lin_reg.coef_))

# # pretty printing coefficients with attributes
# extra_attribs = ['rooms_per_household', 'bedrooms_per_household', 'population_per_household', 'bedrooms_per_rooms']
# cat_one_hot_attribs = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
# attributes = num_attribs + extra_attribs + cat_one_hot_attribs

feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
print(sorted(zip(feature_importances, attributes), reverse=True))

# GRID SEARCH result comparison
cv_results = grid_search.cv_results_
for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
    print(np.sqrt(-mean_score), params)


