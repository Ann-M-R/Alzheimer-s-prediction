# %%
import pickle
import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split   
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

output_file = 'model.bin'


# %%
df = pd.read_csv("alzheimers_disease_data.csv")

# %%
df.head()

# %%
df.columns = df.columns.str.lower()

# %% [markdown]
# ### EDA

# %%
df.columns

# %%
df.dtypes

# %%
df.isnull().sum()

# %%
df.nunique()

# %%
df.columns[df.nunique() == 2]

# %%
df[df.columns[df.nunique() == 2]]

# %%
df.head()

# %%
ethnicity = {
    0: 'caucasian',
    1: 'african american',
    2: 'asian',
    3: 'other'
}

df['ethnicity'] = df['ethnicity'].map(ethnicity)

education_values = {
    0: 'none',
    1: 'high school',
    2: 'bachelor\'s',
    3: 'higher'
}

df['educationlevel'] = df['educationlevel'].map(education_values)


# %%
df.head()

# %%
df.dtypes

# %%
df[df.columns[df.dtypes == 'object']]


# %%
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ','_')

# %%
df[df.columns[df.dtypes == 'object']]

# %%
categorical = ['gender', 'smoking', 'familyhistoryalzheimers', 'cardiovasculardisease',
       'diabetes', 'depression', 'headinjury', 'hypertension',
       'memorycomplaints', 'behavioralproblems', 'confusion', 'disorientation',
       'personalitychanges', 'difficultycompletingtasks', 'forgetfulness', 'ethnicity', 'educationlevel']

# %%
numerical = list(set(df.columns) - set(categorical))
numerical

# %%
numerical.remove('doctorincharge')

# %%
numerical.remove('patientid')

# %%
numerical.remove('diagnosis')

# %%
len(numerical)

# %% [markdown]
# splitting dataset

# %%
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
len(df_full_train), len(df_test)

# %%
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
len(df_train), len(df_val)

# %%
df_train

# %%
df_train = df_train.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)

# %%
df_train

# %%
y_train = df_train.diagnosis.values
y_test = df_test.diagnosis.values
y_val = df_val.diagnosis.values

# %%
del df_train['diagnosis']
del df_test['diagnosis']
del df_val['diagnosis']

# %%
df_full_train


# %%
df_train[categorical + numerical]

# %%
train_dicts =  df_train[categorical + numerical].to_dict(orient = 'records')
dv = DictVectorizer(sparse = False)
X_train = dv.fit_transform(train_dicts)
X_train.shape

# %%
val_dicts =  df_val[categorical + numerical].to_dict(orient = 'records')
X_val = dv.transform(val_dicts)
X_val.shape

# %% [markdown]
# ### Gradient Boosting

print("doing validation")

# %%
features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

# %%
def parse_xgb_output(output):
    results = []

    for line in output.stdout.strip().split('\n'):
        it_line, train_line, val_line = line.split('\t')

        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])

        results.append((it, train, val))
    
    columns = ['num_iter', 'train_auc', 'val_auc']
    df_results = pd.DataFrame(results, columns=columns)
    return df_results

# %%
watchlist = [(dtrain, 'train'), (dval, 'val')]


# %%
#xgb
xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
 
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
 
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}
 
xg_model = xgb.train(xgb_params, dtrain, num_boost_round=175)

# XGBoost Model
y_pred = xg_model.predict(dval)
roc_auc = roc_auc_score(y_val, y_pred)

print(f'auc is {roc_auc}')

#training final model

print("training final model")

y_full_train = df_full_train['diagnosis']
del df_full_train['diagnosis']
del df_full_train['patientid']
del df_full_train['doctorincharge']

dicts_full_train = df_full_train.to_dict(orient='records')
 
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)
 
dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)


feature_names = list(dv.get_feature_names_out())
dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train,
                    feature_names=feature_names)
 
dtest = xgb.DMatrix(X_test, feature_names=feature_names)

y_pred = xg_model.predict(dtest)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'auc = {roc_auc}')

# %%

with open (output_file, 'wb') as f_out:
    pickle.dump((dv,xg_model), f_out) 


print(f'model saved to {output_file}')





