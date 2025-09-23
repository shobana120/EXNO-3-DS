## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
  ```
       import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/e07e69c5-2e11-467e-b9e2-9ce889d68f9e)
# ORDINAL ENCODER
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])

```
# LABEL ENCODER

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(df[["ord_2"]])
dfc
```
![image](https://github.com/user-attachments/assets/19232ab2-535d-4e05-a5e4-a5c569f6ba68)
```
dfc=df.copy()
dfc['con_2']=le.fit_transform(dfc['ord_2'])
dfc
```
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df[['nom_0']]))
enc
```
![image](https://github.com/user-attachments/assets/6a606a27-83ca-4a7b-9ff9-fae20fd5fbd5)
```
df2=pd.concat([df,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/b81c30c5-3d05-41be-bd9f-742528adfd96)
```
pip install --upgrade category_encoders

```
# BinaryEncoder
![image](https://github.com/user-attachments/assets/d02d7da8-1574-46e8-8493-f60a86e42c48)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/522856cb-c7dc-45bf-a8d8-ab427ade90cc)
# TARGET ENCODER
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
# FEATURE ENGINEERING
![image](https://github.com/user-attachments/assets/ab4ae06b-baa4-4b16-afba-f089d4f08e7c)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/39cde9a8-7420-420c-8ff5-c0795f4d4ef1)
```
df["Highly Positive Skew"]=np.log(df["Highly Positive Skew"])
df
```

![image](https://github.com/user-attachments/assets/b60192be-0340-418c-9204-8562ad4af32d)

```
df["Moderate Positive Skew"]=np.reciprocal(df["Moderate Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/4fe9979b-40c8-469b-8654-4c3f87bd9fe0)
# POWER TRANSFORMATION
![image](https://github.com/user-attachments/assets/da6514a4-06fe-4770-a9b4-c5f19e278c26)
```
df["Moderate Negative Skew_yeojohnson"],parameter=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/c1ed44ba-0255-478f-a084-a3579f213557)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/01937ee9-2bda-4b79-b502-0128b30ac2d6)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/1521738c-c23e-4e2a-812b-160ee14cb88e)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/44ff017e-8a7c-4f72-8ccc-633dcd6802c2)

# RESULT:
       # INCLUDE YOUR RESULT HERE

       
