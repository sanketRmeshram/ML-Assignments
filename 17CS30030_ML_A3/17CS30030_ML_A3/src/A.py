import pandas as pd
import numpy as np

df=pd.read_csv("AllBooks_baseline_DTM_Labelled.csv")

for i in range(len(df)):
    df.iloc[i,0]=str(df.iloc[i,0]).split('_')[0]    

df.drop(13,inplace=True)
df.reset_index(inplace=True,drop=True)
df.to_csv("AllBooks_baseline_DTM_Labelled_modified.csv",index=False)  
del df["Unnamed: 0"]


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tf_idf_vector=tfidf_transformer.fit_transform(df)

mat=tf_idf_vector.toarray()

np.save('mat.npy',mat)
