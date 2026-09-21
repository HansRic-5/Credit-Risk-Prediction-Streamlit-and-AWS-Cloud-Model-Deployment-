import pandas as pd
import numpy as np
from typing import Tuple
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

class DataPreprocessor:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.valid_loans = [
            'Auto Loan', 'Credit-Builder Loan', 'Debt Consolidation Loan',
            'Home Equity Loan', 'Mortgage Loan', 'Payday Loan',
            'Personal Loan', 'Student Loan'
        ]

    def _month_converter(self, x):
        if pd.isna(x):
            return np.nan
        parts = str(x).split()
        if len(parts) >= 4:
            return (pd.to_numeric(parts[0], errors='coerce') * 12 + pd.to_numeric(parts[3], errors='coerce'))
        return np.nan

    def _get_loan_counts(self, loan_string):
        if pd.isna(loan_string) or str(loan_string).strip() == '':
            return Counter()
        loans = [item.strip() for item in str(loan_string).replace(' and ', ',').split(',')]
        return Counter([l for l in loans if l in self.valid_loans])

    def clean_and_split(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        df = df.drop(columns=['Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN'], errors='ignore')
        
        mis_col = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 
                   'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly']
        df[mis_col] = df[mis_col].astype(str).replace('_', '', regex=False)
        df[mis_col] = df[mis_col].apply(pd.to_numeric, errors='coerce')
        
        df['Credit_Score'] = df['Credit_Score'].map({'Good': 2, 'Standard': 1, 'Poor': 0})
        
        df = df[df['Payment_Behaviour'] != '!@9#%8']
        
        pattern = r'(High|Low)_spent_(Small|Medium|Large)_value_payments'
        extracted = df['Payment_Behaviour'].str.extract(pattern)
        if not extracted.empty and extracted.shape[1] == 2:
            df[['Spent_Level', 'Payment_Value']] = extracted
        df.drop(columns=['Payment_Behaviour'], inplace=True)
        
        freq_dicts = df['Type_of_Loan'].apply(self._get_loan_counts).tolist()
        encoded_loans_df = pd.DataFrame(freq_dicts, index=df.index).fillna(0).astype(int)
        encoded_loans_df.columns = [f"{col.replace(' ', '_').lower()}_freq" for col in encoded_loans_df.columns]
        for loan in self.valid_loans:
            col_name = f"{loan.replace(' ', '_').lower()}_freq"
            if col_name not in encoded_loans_df.columns:
                encoded_loans_df[col_name] = 0
        df = pd.concat([df, encoded_loans_df], axis=1)
        df.drop(columns=['Type_of_Loan'], inplace=True)
        
        df['Occupation'] = df['Occupation'].replace('_______', 'Unknown')
        df['Credit_Mix'] = df['Credit_Mix'].replace('_', np.nan)
        df['Credit_History_Age'] = df['Credit_History_Age'].apply(self._month_converter).astype('float64')
        
        num_col = df.select_dtypes(exclude='object').drop(columns=['Credit_Score'], errors='ignore').columns
        for col in num_col:
            if col == 'Changed_Credit_Limit':
                continue
            negative_mask = df[col] < 0
            df.loc[negative_mask, col] = np.nan
            
        X = df.drop(columns=['Credit_Score'])
        y = df['Credit_Score']
        
        print("Data cleansing completed. Splitting train/test...")
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)
        
    def get_transformer(self, x_train: pd.DataFrame) -> ColumnTransformer:
        num_feat = x_train.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
        
        spent_level_order = ['Low', 'High']
        payment_value_order = ['Small', 'Medium', 'Large']
        credit_mix_order = ['Bad', 'Standard', 'Good']
        payment_min_order = ['NM', 'Yes', 'No']
        
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])

        ordinal_spent_level = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('Ordinal_Spent', OrdinalEncoder(categories=[spent_level_order], handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        ordinal_payment_value = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('Ordinal_Value', OrdinalEncoder(categories=[payment_value_order], handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        ordinal_Credit_Mix = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('Ordinal_Credit', OrdinalEncoder(categories=[credit_mix_order], handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        ordinal_Payment_Min_Amount = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('Ordinal_Min_Amount', OrdinalEncoder(categories=[payment_min_order], handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        OHE_nominal = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('OHE', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
        
        nominal_cols = [col for col in x_train.select_dtypes(include=['object']).columns if col not in ['Spent_Level', 'Payment_Value', 'Credit_Mix', 'Payment_of_Min_Amount']]
        
        transformers = [
            ('num', numeric_pipeline, num_feat),
        ]
        
        if 'Spent_Level' in x_train.columns:
            transformers.append(('cat_spent', ordinal_spent_level, ['Spent_Level']))
        if 'Payment_Value' in x_train.columns:
            transformers.append(('cat_payment_value', ordinal_payment_value, ['Payment_Value']))
        if 'Credit_Mix' in x_train.columns:
            transformers.append(('cat_credit_mix', ordinal_Credit_Mix, ['Credit_Mix']))
        if 'Payment_of_Min_Amount' in x_train.columns:
            transformers.append(('cat_payment_min', ordinal_Payment_Min_Amount, ['Payment_of_Min_Amount']))
            
        if nominal_cols:
            transformers.append(('cat_nominal', OHE_nominal, nominal_cols))

        return ColumnTransformer(transformers=transformers, remainder='passthrough')
