import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from datetime import *
from sklearn.metrics import mean_absolute_percentage_error
model_FB_path = "modelFB.pkl"
model_Google_path = "modelGoogle.pkl"
pipeline_FB_path = "pipelineFB.pkl"
pipeline_Google_path = "pipelineGoogle.pkl"
TRAIN_DATA_FB_PATH = 'data/trainFB.csv'
TEST_DATA_FB_PATH = 'data/testFB.csv'
TRAIN_DATA_Google_PATH = 'data/trainGoogle.csv'
TEST_DATA_Google_PATH = 'data/testGoogle.csv'

GBP_USD =1.2573907115065885
EUR_USD = 1.0697



class Preprocessing():
    def __init__(self, data, handle_unknown):
        self.handle_unknown = handle_unknown
        self.data = data

    def __Preprocess__(self,TARGET_NAME,type,Google=False):
        if type == "train":

           for i in TARGET_NAME:
                if i in self.columns:
                    self[i] = np.log1p(self.loc[:, [i]])

        else:
            pass
        if type == "train" and Google== False:
            self.drop(columns=['Nom de la campagne', 'Controle', ], inplace=True)

        if Google== False:


            self["end_date"] = pd.to_datetime(self["end_date"], errors='coerce')
            self["end_month"] = self['end_date'].dt.month.astype('int', errors='ignore')
            self['end-year'] = self['end_date'].dt.year.astype('int', errors='ignore')
            self['end_quarter'] = self['end_date'].dt.quarter
            self.drop(columns=['end_date',], inplace=True)
        else:
            pass

        self['start_date'] = pd.to_datetime(self['start_date'], errors='coerce')
        self["start_month"] = self['start_date'].dt.month.astype('int', errors='ignore')
        self['start_day'] = self['start_date'].dt.day.astype('int', errors='ignore')
        self['start_year'] = self['start_date'].dt.year.astype('int', errors='ignore')
        self['start_quarter'] = self['start_date'].dt.quarter
        dates1 = pd.to_datetime({"year": self['start_year'], "month": self['start_month'], "day": self['start_day']})
        self["Is_Weekend"] = dates1.dt.dayofweek > 4
        self.drop(columns=['start_date','start_day',], inplace=True)
        return self


    def _get_categoric_columns(data):
        """Return categoric column names"""
        data_dtypes_dict = dict(data.dtypes)
        categorical_columns = data.select_dtypes(exclude='number').columns
        return categorical_columns

    def _get_numeric_columns(data):
        """Return numeric column names"""
        data_dtypes_dict = dict(data.dtypes)
        numerical_columns = data.select_dtypes(exclude=["object", "bool"]).columns
        return numerical_columns

class Preprocessing_Google(Preprocessing):
    def __init__(self):
        super().__init__(self)

    def get_cost(data):
        Conv_list = []
        for i in range(len(data['Currency code'])):
            Conv_list.append(convert_toUSD(data['Currency code'][i], data['Cost'][i]))
        return Conv_list

    def get_date(data):
        Start_date_lst, End_date_lst = [], []
        for day in data['Week']:
            dt = datetime.strptime(day.strftime('%Y-%m-%d'), '%Y-%m-%d')
            start = dt - timedelta(days=dt.weekday())
            end = start + timedelta(days=6)
            Start_date_lst.append(start)
            End_date_lst.append(end)
        return Start_date_lst, End_date_lst

    def get_Impressions(self):
        Impressions = self['Impr.'].astype(str)
        Impression_ = []
        for i in Impressions:
            Impression_.append(i.split('.'))
        Impression_list = []
        for i in Impression_:
            if i[1] == '0':
                # print('integer')
                Impression_list.append(int(i[0]))
            else:
                # print(len(i[1]))
                if len(i[1]) == 1:
                    Impression_list.append(int(i[0]) * 1000 + int(i[1]) * 100)
                elif len(i[1]) == 2:
                    Impression_list.append(int(i[0]) * 1000 + int(i[1]) * 10)
                elif len(i[1]) == 3:
                    Impression_list.append(int(i[0]) * 1000 + int(i[1]))
        return Impression_list

    def __Preprocess__(self,TARGET_NAME,type):
        if type=="train":
             self.loc[:, "amount"] = Preprocessing_Google.get_cost(self)
             self['start_date'] = Preprocessing_Google.get_date(self)[0]
             self['impressions'] = Preprocessing_Google.get_Impressions(self)
             self.rename(columns={'Views': 'views'}, inplace=True)
             self.rename(columns={'Clicks': 'clicks'}, inplace=True)
             self = self[['Campaign type', 'clicks', 'impressions', 'amount', 'start_date','views']]
        else:
            pass
        Preprocessing.__Preprocess__(self,TARGET_NAME,type,Google=True)
        return self



class Spliter(Preprocessing):
    def __init__(self, data):
        self.Data = data
    def convert_cat(self):
        cols = self.columns.values.tolist()

        for col in cols:
            try:
                 self[col] = pd.to_numeric(self[col])
            except:
                 continue
        return self

    def __Spliter__(Data, TARGET_NAME):

        cols = Data.columns.values.tolist()
        for i in TARGET_NAME:
            cols.remove(i)

        Data_cols = cols.copy()
        for i in TARGET_NAME:
            cols.append(i)
        print(cols)
        Data.dropna(inplace=True)
        y = Data.loc[:, TARGET_NAME]
        X = Data.loc[:, Data_cols]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=33)
        TRAIN_DATA = pd.DataFrame(np.concatenate([X_train, y_train], axis=1), columns=cols)
        TEST_DATA = pd.DataFrame(np.concatenate([X_test, y_test], axis=1), columns=cols)
        TRAIN_DATA = Spliter.convert_cat(TRAIN_DATA)
        TEST_DATA = Spliter.convert_cat(TEST_DATA)
        return [TRAIN_DATA, TEST_DATA]

class Imputer(TransformerMixin):
    """
    Median and Mode imputer
    """

    def __init__(self, strategy):
        self.strategy = strategy

    @staticmethod
    def to_lowercase(x):
        if pd.isnull(x):
            return x
        return x.lower()

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : pandas.DataFrame
        y : ignored
        Returns
        -------
        self : object
        """
        if self.strategy == 'median':
            self.numerical_columns = list(X.select_dtypes('number').columns)
            self.median_dict_ = {}

            if len(self.numerical_columns) == 0:
                return self
            for col in self.numerical_columns:
                self.median_dict_[col] = X[col].median()

        elif self.strategy == 'mode':
            self.categorical_columns = list(X.select_dtypes('object').columns)
            self.mode_dict_ = {}
            if len(self.categorical_columns) == 0:
                return self

            for col in self.categorical_columns:
                """Lowercase and fit"""
                X.loc[:, col] = X[col].map(Imputer.to_lowercase)
                self.mode_dict_[col] = X[col].mode().iloc[0]
        return self

    def transform(self, X):

        """
        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        X : pandas.DataFrame
        Imputed data
        """
        if self.strategy == 'median':
            if len(self.numerical_columns) == 0:
                return X
            for col in self.numerical_columns:
                X[col].fillna(self.median_dict_[col], inplace=True)

        elif self.strategy == 'mode':
            if len(self.categorical_columns) == 0:
                return X
            for col in self.categorical_columns:
                """Lowercase and impute"""
                X.loc[:, col] = X[col].map(Imputer.to_lowercase)
                X[col].fillna(self.mode_dict_[col], inplace=True)
        return X


class CustomRobustScaler(RobustScaler):
    def __init__(self, **params):
        super().__init__(**params)
    def _get_original_data(self, X, scaled_data):
        """Concatenate numerical and
        categorical columns"""
        X_numeric_data = pd.DataFrame(scaled_data, columns=self.numerical_columns)
        X_remnant_data = X.drop(self.numerical_columns, axis=1)
        X_original = pd.concat([X_numeric_data.reset_index(drop=True), X_remnant_data.reset_index(drop=True)], axis=1)
        X_original = X_original.sort_index(axis=1)
        return X_original

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : pandas.DataFrame
        y : ignored

        Returns
        -------
        self : object
        """
        self.numerical_columns = list(X.select_dtypes(['number']).columns)
        print(f"Num COls: {self.numerical_columns}")
        if len(self.numerical_columns) == 0:
            """No numerical columns detected"""
            return self

        X_numeric = X[self.numerical_columns]
        super().fit(X_numeric)
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : pandas.DataFrame

        Returns
        -------
        X_original : scaled data
        """

        if len(self.numerical_columns) == 0:
            return X

        X_numeric = X[self.numerical_columns]
        scaled_data = super().transform(X_numeric)
        X_original = self._get_original_data(X, scaled_data)
        return X_original


def fit_model(model, X_train, y_train,X_test,y_test):
    model = model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_MSE = mean_squared_error(y_train, y_train_pred)
    test_MSE = mean_squared_error(y_test, y_test_pred)
    train_RMSE = np.sqrt(train_MSE)
    test_RMSE = np.sqrt(test_MSE)
    train_MAPE = mean_absolute_percentage_error(y_train, y_train_pred) * 100
    test_MAPE = mean_absolute_percentage_error(y_test, y_test_pred) * 100
    train_R2 = r2_score(y_train, y_train_pred)
    test_R2 = r2_score(y_test, y_test_pred)
    Evaluation_dict = {
        'model': model,
        "model_name": model.__class__.__name__,
        "train_RMSE":train_RMSE,
        "test_RMSE": test_RMSE,
        "train_MAPE": train_MAPE,
        "test_MAPE": test_MAPE,
        "train_R2": train_R2,
        "test_R2": test_R2
    }
    return Evaluation_dict
def fit_Google_model(model, X_train, y_train,X_test,y_test):

    model = MultiOutputRegressor(estimator=model).fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_MSE = mean_squared_error(y_train, y_train_pred)
    test_MSE = mean_squared_error(y_test, y_test_pred)
    train_RMSE = np.sqrt(train_MSE)
    test_RMSE = np.sqrt(test_MSE)
    train_MAPE = mean_absolute_percentage_error(y_train, y_train_pred) * 100
    test_MAPE = mean_absolute_percentage_error(y_test, y_test_pred) * 100
    train_R2 = r2_score(y_train, y_train_pred)
    test_R2 = r2_score(y_test, y_test_pred)
    Evaluation_dict = {
        'model': model,
        "model_name": model.__class__.__name__,
        "train_RMSE":train_RMSE,
        "test_RMSE": test_RMSE,
        "train_MAPE": train_MAPE,
        "test_MAPE": test_MAPE,
        "train_R2": train_R2,
        "test_R2": test_R2
    }
    return Evaluation_dict
def convert_toUSD(actual, value):
     if actual == 'USD':
        return value
     elif actual == 'EUR':
        return round(value * EUR_USD, 2)
     elif actual == 'GBP':
        return round(value * GBP_USD, 2)
def fit_Google_model(model, X_train, y_train,X_test,y_test):

    model = MultiOutputRegressor(estimator=model).fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_MSE = mean_squared_error(y_train, y_train_pred)
    test_MSE = mean_squared_error(y_test, y_test_pred)
    train_RMSE = np.sqrt(train_MSE)
    test_RMSE = np.sqrt(test_MSE)
    train_MAPE = mean_absolute_percentage_error(y_train, y_train_pred) * 100
    test_MAPE = mean_absolute_percentage_error(y_test, y_test_pred) * 100
    train_R2 = r2_score(y_train, y_train_pred)
    test_R2 = r2_score(y_test, y_test_pred)
    Evaluation_dict = {
        'model': model,
        "model_name": model.__class__.__name__,
        "train_RMSE":train_RMSE,
        "test_RMSE": test_RMSE,
        "train_MAPE": train_MAPE,
        "test_MAPE": test_MAPE,
        "train_R2": train_R2,
        "test_R2": test_R2
    }
    return Evaluation_dict
