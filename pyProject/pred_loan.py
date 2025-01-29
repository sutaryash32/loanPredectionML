import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')  


df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)


X = df.drop(['Loan_ID', 'LoanAmount'], axis=1) 
y = df['LoanAmount']  


train_data = df[df['LoanAmount'].notna()]
predict_data = df[df['LoanAmount'].isna()]

X_train = train_data.drop(['Loan_ID', 'LoanAmount'], axis=1)
y_train = train_data['LoanAmount']
X_predict = predict_data.drop(['Loan_ID', 'LoanAmount'], axis=1)


num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term', 'Credit_History']
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']


num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])


X_train_preprocessed = preprocessor.fit_transform(X_train)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_preprocessed, y_train)


X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train_preprocessed, y_train, test_size=0.2, random_state=42)
model.fit(X_train_split, y_train_split)
y_pred = model.predict(X_test_split)
print(f'RMSE: {np.sqrt(mean_squared_error(y_test_split, y_pred)):.2f}')
print(f'MAE: {mean_absolute_error(y_test_split, y_pred):.2f}')


if not X_predict.empty:
    X_predict_preprocessed = preprocessor.transform(X_predict)
    predicted_loan_amounts = model.predict(X_predict_preprocessed)
    df.loc[df['LoanAmount'].isna(), 'LoanAmount'] = predicted_loan_amounts


print(df[['Loan_ID', 'LoanAmount']])