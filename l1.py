import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import fetch_california_housing

data=fetch_california_housing(as_frame=True)
housing_df=data.frame

numerical_features=housing_df.select_datatypes(includes=[np.number]).columns

plt.figure(figsize=(15,10))
for i, feature in enumerate(numerical_features):
    plt.subplot(3,3,i+1)
    sns.histplot(housing_df[feature],kde=True,bin=30,color="blue")
    plt.title("distribution of {feature}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,10))
for i, feature in enumerate(numerical_features):
    plt.subplot(3,3,i+1)
    sns.boxplot(x=housing_df[feature],color="blue")
    plt.title("box distribution of {feature}")
plt.tight_layout()
plt.show()

print("outliers detection")
outliers_summary={}
for feature in numerical_features:
    q1=housing_df[feature].quantile(0.25)
    q3=housing_df[feature].quantile(0.75)
    IQR=q3-q1
    lower_bound=q1-1.5*IQR
    upper_bound=q3+1.5*IQR
    outliers=housing_df(housing_df[feature] < lower_bound | housing_df[feature] > upper_bound)
    outliers_summary=len(outliers)
    print("{feature}:len(outliers) outliers")

    