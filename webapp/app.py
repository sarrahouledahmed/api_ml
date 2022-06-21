from utils import *
import pandas as pd
import xgboost
import lightgbm
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from flask import Flask,request, render_template, jsonify
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from utils import Preprocessing
import re
cols = ['sector', 'objective', 'amount','start_date','end_date']
cols_Google = ['Campaign type', 'amount','start_date']
TARGET_FB_NAME=['result','reach','impressions']
TARGET_Google_NAME=['clicks','impressions','views']

app = Flask(__name__)
@app.route('/')
def home():
    return render_template("index.html")

@app.route("/train_Digital_Marketing_campaign_model")
def training():
    train_data = pd.read_csv(TRAIN_DATA_FB_PATH)
    test_data = pd.read_csv(TEST_DATA_FB_PATH)
    x_train, y_train = (train_data.drop(TARGET_FB_NAME,axis=1),train_data[TARGET_FB_NAME])
    x_test, y_test = (test_data.drop(TARGET_FB_NAME, axis=1),test_data[TARGET_FB_NAME])
    x_train= x_train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    x_test= x_test.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    if 'Unnamed0' in x_test.columns:
        x_test.drop(columns=['Unnamed0'],axis=1,inplace=True)

    if 'Unnamed0' in x_train.columns:
        x_train.drop(['Unnamed0'],axis=1,inplace=True)

    numeric_features = Preprocessing._get_numeric_columns(x_train)
    categorical_features= Preprocessing._get_categoric_columns(x_train)
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
            ('numericals', numeric_transformer, numeric_features)
        ],
        remainder='drop'
    )
    print(x_train.columns)
    x_train_preprocessed=preprocessor.fit_transform(x_train)
    x_test_preprocessed = preprocessor.transform(x_test)
    xgboost_ = xgboost.XGBRegressor(colsample_bytree=0.7, learning_rate=0.05, max_depth=5, min_child_weight=4,
                                    objective='reg:linear', nthread=3, n_estimators=300, silent=1, subsample=0.7)
    CatBoost_ = CatBoostRegressor(n_estimators=300, loss_function='RMSE', learning_rate=0.05, depth=5,
                                  task_type='CPU', random_state=1, verbose=False)
    lightgbm_ = lightgbm.LGBMRegressor(boosting_type='gbdt', max_depth=5, n_estimators=300,
                                       learning_rate=0.05, objective='rmse', n_jobs=2, random_state=0)
    DecisionTree_ = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=1)
    RMSE_List=[]
    models = [
        DecisionTree_,lightgbm_,
        CatBoost_,xgboost_
        ]
    for model in models:
        Evaluation_dict = fit_Google_model(model, x_train_preprocessed, y_train, x_test_preprocessed, y_test)
        RMSE_List.append(Evaluation_dict)
        RMSE_df = pd.DataFrame(RMSE_List).sort_values("test_RMSE", ascending=True)
    model = RMSE_df.iloc[0]['model']
    print(RMSE_df['train_RMSE'], RMSE_df['test_RMSE'])
    test_RMSE = RMSE_df.iloc[0]['test_RMSE']
    train_RMSE = RMSE_df.iloc[0]['train_RMSE']
    joblib.dump(model, open(model_FB_path, 'wb'))
    joblib.dump(preprocessor, open(pipeline_FB_path, 'wb'))

    return {
            "status": "success",
            "test_rmse": f"{test_RMSE}",
            "train_RMSE": f"{train_RMSE}"
        }

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    score_df= pd.DataFrame([final], columns = cols)
    try:
        model = joblib.load(open(model_FB_path, 'rb'))
        pipeline = joblib.load(open(pipeline_FB_path, 'rb'))
    except Exception as e:
        print(f"Exception: {e}")
        return {
            'status': "failure",
            "message": "Train the model first"
        }
    #Saving the log
    score_df.to_csv("new1.csv",index=False)
    df = pd.read_csv(r'new1.csv')
    df= Preprocessing.__Preprocess__(df,TARGET_NAME=TARGET_FB_NAME,type="test")
    df= df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    x_test_preprocessed = pipeline.transform(df)
    predictions = np.expm1(model.predict(x_test_preprocessed))
    result = int(round(predictions[0][0], 2))
    reach= int(round(predictions[0][1], 2))
    impressions= round(predictions[0][2])
    upper_band_result = result + 0.25 * result
    lower_band_result = result- 0.25 * result
    upper_band_reach = reach + 0.25 * reach
    lower_band_reach = reach - 0.25 * reach
    upper_band_impressions = impressions + 0.25 * impressions
    lower_band_impressions = impressions - 0.25 * impressions
    result_margin= np.array([int(lower_band_result), int(upper_band_result)])
    reach_margin = np.array([int(lower_band_reach), int(upper_band_reach)])
    impressions_margin= np.array([lower_band_impressions, int(upper_band_impressions)])

    return render_template('index.html', prediction_text="the campaign Result should be between" +str(result_margin)+"\n Predicted Impressions should be between: \n"+str(impressions_margin)+"\n Predicted reach should be between: \n"+str(reach_margin))

@app.route("/training_Google")
def training_Google():
    TRAIN_DATA = pd.read_csv(TRAIN_DATA_Google_PATH)
    TEST_DATA = pd.read_csv(TEST_DATA_Google_PATH)
    x_train, y_train = (TRAIN_DATA.drop(TARGET_Google_NAME, axis=1), TRAIN_DATA[TARGET_Google_NAME])
    x_test, y_test = (TEST_DATA.drop(TARGET_Google_NAME, axis=1), TEST_DATA[TARGET_Google_NAME])
    print(x_train.columns)
    print(y_train.columns)

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
            ('numericals', numeric_transformer, numeric_features)
        ],
        remainder='drop'
    )
    x_train_preprocessed = preprocessor.fit_transform(x_train)
    x_test_preprocessed = preprocessor.transform(x_test)
    xgboost_ = xgboost.XGBRegressor(colsample_bytree=0.7, learning_rate=0.05, max_depth=3, min_child_weight=4,
                                    objective='reg:linear', nthread=4, n_estimators=100, silent=1, subsample=0.7)
    CatBoost_ = CatBoostRegressor(n_estimators=100, loss_function='RMSE', learning_rate=0.05, depth=3,
                                  task_type='CPU', random_state=1, verbose=False)
    lightgbm_ = lightgbm.LGBMRegressor(boosting_type='gbdt', max_depth=3, n_estimators=100,
                                       learning_rate=0.05, objective='rmse', n_jobs=3, random_state=0)
    DecisionTree_ = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=1)
    RMSE_List = []
    models = [DecisionTree_, lightgbm_,
              CatBoost_, xgboost_
              ]
    for model in models:
        Evaluation_dict = fit_Google_model(model, x_train_preprocessed, y_train, x_test_preprocessed, y_test)
        RMSE_List.append(Evaluation_dict)
        RMSE_df = pd.DataFrame(RMSE_List).sort_values("test_RMSE", ascending=True)
    model = RMSE_df.iloc[0]['model']
    print(RMSE_df['train_RMSE'],RMSE_df['test_RMSE'])
    test_RMSE = RMSE_df.iloc[0]['test_RMSE']
    train_RMSE = RMSE_df.iloc[0]['train_RMSE']
    joblib.dump(model, open(model_Google_path, 'wb'))
    joblib.dump(preprocessor, open(pipeline_Google_path, 'wb'))

    return {
        "status": "success",
        "test_rmse": f"{test_RMSE}",
        "train_RMSE": f"{train_RMSE}"
            }
@app.route('/predict_Google',methods=['POST','GET'])
def predict_Google():
    if request.method == 'POST':
          init_features = [x for x in request.form.values()]
          final_features = np.array([init_features])
          score_df = pd.DataFrame(final_features,columns=cols_Google)
          print(score_df)

    if request.method == 'GET':
        return render_template('index_GOOGLE.html' )


    try:
        model = joblib.load(open(model_Google_path, 'rb'))
        pipeline = joblib.load(open(pipeline_Google_path, 'rb'))
    except Exception as e:
        print(f"Exception: {e}")

        return {
            'status': "failure",
            "message": "Train the model first"
        }

    ###Saving the log
    score_df.to_csv("new2.csv",index=False)
    df = pd.read_csv(r'new2.csv')
    df= Preprocessing_Google.__Preprocess__(df,TARGET_NAME=TARGET_Google_NAME,type="test")
    x_test_preprocessed = pipeline.transform(df)
    predictions = np.expm1(model.predict(x_test_preprocessed))
    clicks = int(round(predictions[0][0], 2))
    impressions= int(round(predictions[0][1], 2))
    views= int(round(predictions[0][1], 2))
    upper_band_clicks = clicks + 0.25 * clicks
    lower_band_clicks = clicks - 0.25 * clicks
    upper_band_views = views + 0.25 * views
    lower_band_views = views - 0.25 * views
    upper_band_impressions = impressions + 0.25 * impressions
    lower_band_impressions = impressions - 0.25 * impressions
    clicks_margin= np.array([int(lower_band_clicks), int(upper_band_clicks)])
    views_margin = np.array([int(lower_band_views), int(upper_band_views)])
    impressions_margin= np.array([lower_band_impressions, int(upper_band_impressions)])

    return render_template('index_GOOGLE.html', prediction_text="Predicted Clicks should be between:" + str(clicks_margin)+"\n Predicted Impressions should be between: \n"+str(impressions_margin)+"\n Predicted Views should be between: \n"+str(views_margin) )

if __name__ == "__main__":
    app.run(host='localhost', port=5000)

