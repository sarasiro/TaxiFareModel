# imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn import set_config; set_config(display='diagram')
from sklearn.base import BaseEstimator,  TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from TaxiFareModel.encoders import TimeFeaturesEncoder,DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data,clean_data

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipe_distance = make_pipeline(DistanceTransformer(), RobustScaler())
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']

        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), StandardScaler())
        time_cols = ['pickup_datetime']

        feat_eng_bloc = ColumnTransformer([('time', pipe_time, time_cols),('distance', pipe_distance, dist_cols)])

        pipe_cols = Pipeline(steps=[('feat_eng_bloc', feat_eng_bloc), ('LinearRegression', LinearRegression())])

        return pipe_cols

    def run(self, X_train, y_train, pipeline):
        """set and train the pipeline"""
        pipeline.fit(X_train, y_train)
        return pipeline

    def evaluate(self, X_test, y_test,pipeline):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    X = df.drop(columns='fare_amount')
    y = df['fare_amount']
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # train
    trainer=Trainer(X,y)
    pipe = trainer.set_pipeline()

    pipe = trainer.run(X_train, y_train, pipe)
    # evaluate
    evaluate=trainer.evaluate(X_test, y_test, pipe)

    print(evaluate)

    print('TODO')
