from utils import *
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost
import lightgbm
import joblib
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from utils import Preprocessing_Google,Preprocessing
cols = ['Campaign type', 'Montant dépensé (USD)','Commence','Fin']
TARGET_NAME=['Clicks','Impressions']
TRAIN_DATA = pd.read_csv(TRAIN_DATA_Google_PATH)
TEST_DATA = pd.read_csv(TRAIN_DATA_Google_PATH)
x_train, y_train = (TRAIN_DATA.drop(TARGET_NAME,axis=1),TRAIN_DATA[TARGET_NAME])
x_test, y_test = (TEST_DATA.drop(TARGET_NAME, axis=1),TEST_DATA[TARGET_NAME])
numeric_features = Preprocessing._get_numeric_columns(x_train)
categorical_features = Preprocessing._get_categoric_columns(x_train)
categorical_transformer = Pipeline(
    [('imputer_cat', SimpleImputer(strategy='constant')),
     ('onehot', OneHotEncoder(handle_unknown='ignore'))]
)
numeric_transformer = Pipeline(
    [
        ('imputer_num', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)
preprocessor = ColumnTransformer(
    [
        ('categoricals', categorical_transformer, categorical_features),
        ('nume'
         'ricals', numeric_transformer, numeric_features)
    ],
    remainder='drop'
)
x_train_preprocessed=preprocessor.fit_transform(x_train)
x_test_preprocessed = preprocessor.transform(x_test)
xgboost_ =xgboost.XGBRegressor(colsample_bytree=0.7, learning_rate=0.05, max_depth=3, min_child_weight=4,
                                    objective='reg:linear', nthread=4, n_estimators=100, silent=1, subsample=0.7)
CatBoost_ = CatBoostRegressor(n_estimators=200, loss_function='RMSE', learning_rate=0.05, depth=4,
                                  task_type='CPU',random_state=1, verbose=False)
lightgbm_ =lightgbm.LGBMRegressor(boosting_type='gbdt', max_depth=3,n_estimators=200,
                                       learning_rate=0.03, objective='rmse', n_jobs=2, random_state=0)
DecisionTree_ = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=1)
RMSE_List=[]
models = [DecisionTree_,lightgbm_,
        CatBoost_,xgboost_
        ]
for model in models:
     Evaluation_dict = fit_model_Google(model,x_train_preprocessed, y_train,x_test_preprocessed,y_test)
     RMSE_List.append(Evaluation_dict)
     RMSE_df = pd.DataFrame(RMSE_List).sort_values("test_RMSE", ascending=True)
model = RMSE_df.iloc[0]['model']
RMSE = RMSE_df.iloc[0]['test_RMSE']
joblib.dump(model, open(model_Google_path, 'wb'))
joblib.dump(preprocessor, open(pipeline_Google_path, 'wb'))
print("status : success","selected_model:" ,model,"RMSE:", RMSE)
