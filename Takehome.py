#Reading data and selecting columns
import pandas as pd

df=pd.read_csv('Q2_20230202_majority.csv')
df=df[['tweet','label_majority']]
df = df.dropna(subset=["tweet", "label_majority"]).reset_index(drop=True)

