import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        path = os.path.dirname(__file__)
        award = pd.read_csv(os.path.join(path, 'award_notices_RAMP.csv.zip'),
                            compression='zip', low_memory=False)
        # obtain features from award
        award['Name_processed'] = award['incumbent_name'].str.lower()
        award['Name_processed'] = \
            award['Name_processed'].str.replace(r'[^\w]', '')
        award_features = \
            award.groupby(['Name_processed'])['amount'].agg(['count', 'sum'])

        def zipcodes(X):
            zipcode_nums = pd.to_numeric(X['Zipcode'], downcast='integer', 
                                        errors='coerce')
                                        
            val_prefix = zipcode_nums.values[:, np.newaxis]/1000
            val_suffix = zipcode_nums.values[:, np.newaxis]%1000
            return np.concatenate((val_prefix,val_suffix),axis=1)

        zipcode_transformer = FunctionTransformer(zipcodes, validate=False)

        numeric_transformer = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='mean'))])

        def process_numeric_APE(X):
            APE_prefix = X['Activity_code (APE)'].str[:2]
            APE_root = X['Activity_code (APE)'].str[2]
            val_prefix = pd.to_numeric(APE_prefix, 
                                    errors='coerce').values[:, np.newaxis]
            val_root = pd.to_numeric(APE_root, 
                                    errors='coerce').values[:, np.newaxis]
            return np.concatenate((val_prefix,val_root),axis=1)

        APE_n_transformer = FunctionTransformer(process_numeric_APE, 
                                    validate=False) 

        def process_date_ordinal(X):
            date = pd.to_datetime(X['Fiscal_year_end_date'], format='%Y-%m-%d')
            ord_date = date.apply(lambda x: x.toordinal()).values[:,np.newaxis]
            return ord_date - pd.Timestamp('2013-01-01').toordinal()

        date_ord_transformer = FunctionTransformer(process_date_ordinal, validate=False)

        def merge_naive(X):
            X['Name'] = X['Name'].str.lower()
            X['Name'] = X['Name'].str.replace(r'[^\w]', '')
            df = pd.merge(X, award_features, left_on='Name',
                          right_on='Name_processed', how='left')
            return df[['count', 'sum']]
        merge_transformer = FunctionTransformer(merge_naive, validate=False)

        def merge_bid(X):
            sub_award = award[['Buyer_name','Total_amount']]
            sub_award['Name_processed'] = sub_award['Buyer_name'].str.lower()
            sub_award['Name_processed'] = sub_award['Name_processed'].str.replace('[^\w]','')

            sub_award_features = sub_award.groupby(['Name_processed'])['Total_amount'].agg(['count','sum'])

            sub_X = X[['Name']]
            sub_X['Name'] = sub_X['Name'].str.lower()     
            sub_X['Name'] = sub_X['Name'].str.replace('[^\w]','')
            df = pd.merge(sub_X, sub_award_features, left_on=['Name'], right_on=['Name_processed'], how='left')

            return df[['count','sum']]

        merge_bid_transformer = FunctionTransformer(merge_bid, validate=False)

        def merge_bid_year(X):
            sub_award = award[['Buyer_name','Publication_date','Total_amount']]
            sub_award['Name_processed'] = sub_award['Buyer_name'].str.lower()
            sub_award['Name_processed'] = sub_award['Name_processed'].str.replace('[^\w]','')
            sub_award['Publication_date'] = pd.to_datetime(sub_award['Publication_date'],errors='coerce').dt.year

            sub_award_features = sub_award.groupby(['Name_processed','Publication_date'])['Total_amount'].agg(['count','sum'])

            sub_X = X[['Name','Year']]
            sub_X['Name'] = sub_X['Name'].str.lower()     
            sub_X['Name'] = sub_X['Name'].str.replace('[^\w]','')
            df = pd.merge(sub_X, sub_award_features, left_on=['Name','Year'], right_on=['Name_processed','Publication_date'], how='left')

            return df[['count','sum']]

        merge_bid_year_transformer = FunctionTransformer(merge_bid_year, validate=False)


        reg = ExtraTreesRegressor(n_estimators=10, random_state=0)
        merge_name_col = ['Name']
        merge_name_date_col = ['Name','Year']

        num_cols = ['Legal_ID', 'Headcount']
        zipcode_col = ['Zipcode']
        date_cols = ['Fiscal_year_end_date']
        drop_cols = ['Name', 'Address', 'City','Fiscal_year_duration_in_months','Year']
        APE_col = ['Activity_code (APE)']
        merge_col = ['Name']

        preprocessor = ColumnTransformer(
            transformers=[
                ('zipcode', make_pipeline(zipcode_transformer, SimpleImputer(strategy='constant', fill_value=0)), zipcode_col),
                ('num', numeric_transformer, num_cols),
                ('date', make_pipeline(date_ord_transformer, SimpleImputer(strategy='mean')), date_cols),
                ('APE_n', make_pipeline(APE_n_transformer, SimpleImputer(strategy='constant', fill_value=0)), APE_col),
                ('merge', make_pipeline(merge_transformer, SimpleImputer(strategy='median')), merge_col),
                ('merge_bid', make_pipeline(merge_bid_transformer, SimpleImputer(strategy='median')), merge_name_col),
                ('merge_bid_year', make_pipeline(merge_bid_year_transformer, IterativeImputer(random_state=0, estimator=reg)), merge_name_date_col),
                ('drop cols', 'drop', drop_cols),
            ])

        self.preprocessor = preprocessor
        self.preprocessor.fit(X_df, y_array)
        return self

    def transform(self, X_df):
        return self.preprocessor.transform(X_df)
