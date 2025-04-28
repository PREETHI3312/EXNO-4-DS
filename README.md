# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
     # FEATURE SCALING

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/6c93d68a-7906-41d8-8cf1-093a74a6f1a9)
```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/6582ffaf-bbe6-4cb4-aa1f-71dc4f76889d)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/2226a815-0be2-4da1-8def-4fe771ba9066)
```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
# This is typically used in feature scaling,
#particularly max-abs scaling, which is useful
#when you want to scale data to the range [-1, 1]
#while maintaining sparsity (often used with sparse data).
```

![image](https://github.com/user-attachments/assets/edf8706e-266e-48f9-a0f0-531fafa4885f)
```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```

![image](https://github.com/user-attachments/assets/8eff4200-273c-4471-a197-6209976bb1cf)
```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```

![image](https://github.com/user-attachments/assets/dcf2ad6a-5e62-40b3-867d-94151fc239e2)
```
#MIN-MAX SCALING:
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

![image](https://github.com/user-attachments/assets/62f5bc9e-6ae8-4466-945d-4e4ed37d52c0)
```
#MAXIMUM ABSOLUTE SCALING:

from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

![image](https://github.com/user-attachments/assets/73c3e283-cf7a-457f-bd47-981e67aca354)
```
#ROBUST SCALING

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```

![image](https://github.com/user-attachments/assets/538d43d0-510a-4858-87a6-3d4510c42e1f)
```
#FEATURE SELECTION:

df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```

![image](https://github.com/user-attachments/assets/84a3f2ca-c438-4623-88b7-f325a7c03ea8)
```
df_null_sum=df.isnull().sum()
df_null_sum
```

![image](https://github.com/user-attachments/assets/1c236ed9-8da4-4bc7-a61e-197e9e4e715c)
```
# Chi_Square
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
#In feature selection, converting columns to categorical helps certain algorithms
# (like decision trees or chi-square tests) correctly understand and
 # process non-numeric features. It ensures the model treats these columns as categories,
  # not as continuous numerical values.
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/60761d89-76f0-4030-9532-cd6e26d52477)
```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
##This code replaces each categorical column in the DataFrame with numbers that represent the categories.
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/21c478c8-8661-4f7c-a659-ef67dfc6da24)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

```

![image](https://github.com/user-attachments/assets/0d2e1dc2-9e2a-4cbf-a862-8aa6fcd252ca)
```
y_pred = rf.predict(X_test)
df=pd.read_csv("/content/income(1) (1).csv")
df.info()

```

![image](https://github.com/user-attachments/assets/2cf6f44a-707b-4acc-ba67-054d527e58aa)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/2e2d41c7-a3c4-4fe7-aae6-53059dc97b2f)
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/20796eea-c30c-42dd-be49-0c9440f42c53)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

![image](https://github.com/user-attachments/assets/7d73efbf-6262-438b-b167-1f99f6baf041)
```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```

![image](https://github.com/user-attachments/assets/4afd752f-3fb0-468a-94a4-18db98c1b234)
```
!pip install skfeature-chappers
```
```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
```
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

df[categorical_columns] = df[categorical_columns].astype('category')
```
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/74a5dd76-b99a-4f4c-8f67-687357ceb4e6)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
```
```
selected_features_anova = X.columns[selector_anova.get_support()]
```
```
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```

![image](https://github.com/user-attachments/assets/e905bbe0-f9ff-4a25-a34b-acbc105c7c5e)
```
# Wrapper Method
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")
# List of categorical columns
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

# Convert the categorical columns to category dtype
df[categorical_columns] = df[categorical_columns].astype('category')
```
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
```
```
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/69eec235-349f-4237-9573-6139baf5bfd4)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
logreg = LogisticRegression()
n_features_to_select =6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```

![image](https://github.com/user-attachments/assets/404be47f-43f3-48ff-bf3f-0495bee335da)


# RESULT:
  Thus reading  the given data and performing Feature Scaling and Feature Selection process and saving the
data to the file is done successfully.
       
