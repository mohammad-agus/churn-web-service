import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold


C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'


df = pd.read_csv('data_week_3.csv')

df.columns = [i.lower() for i in df.columns]

for col in df.columns[df.dtypes == 'object']:
    df[col] = df[col].str.lower().str.replace(" ", "_")

df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
df['totalcharges'] = df['totalcharges'].fillna(0)

categorical = list(df.dtypes[df.dtypes.values == 'object'].index)
categorical.remove('churn')
categorical.remove('customerid')
categorical.append('seniorcitizen')

numerical = list(df.columns[df.dtypes != 'object'])
numerical.remove('seniorcitizen')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

def train(df_X_train, y_train, C):
    dv = DictVectorizer(sparse=False)

    dict_df_X_train = df_X_train.to_dict(orient='records')
    X_train = dv.fit_transform(dict_df_X_train)

    logregmodel = LogisticRegression(max_iter=3000, C=C)
    logregmodel.fit(X_train, y_train)

    return dv, logregmodel

def predict(dv, model, df_X):
    dict_df_X = df_X.to_dict(orient='records')
    X = dv.transform(dict_df_X)

    y_pred = model.predict_proba(X)[:, 1]
    return y_pred

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

print(f'Doing validation with C = {C}')

roc_auc_scores = []
fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = (df_train['churn'] == 'yes').astype(int).values
    y_val = (df_val['churn'] == 'yes').astype(int).values

    df_train = df_train[numerical+categorical]
    df_val = df_val[numerical+categorical]

    dv, logregmodel = train(df_X_train=df_train, y_train=y_train, C=C)
    y_pred = predict(dv=dv, model=logregmodel, df_X=df_val)

    auc = roc_auc_score(y_true=y_val, y_score=y_pred)
    roc_auc_scores.append(auc)

    print(f'auc on fold {fold} is {auc}')
    fold += 1

print(f'ROC AUC Scores mean: {round(np.mean(roc_auc_scores),3)} +- {round(np.std(roc_auc_scores),3)}')


with open(output_file, 'wb') as f:
    pickle.dump((dv, logregmodel), f)

print(f'The encoder and the model are saved to {output_file}')

