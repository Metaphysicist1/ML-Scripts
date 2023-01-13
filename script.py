import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


df = pd.read_csv('wine-clustering.csv')

#print(df.head())

shape = df.shape
columns = df.columns
print(shape,'\n',columns)

model = KMeans(df,3)
model.fit()

