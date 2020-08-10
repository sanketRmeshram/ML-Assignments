import pandas as pd
import math

#####################Dataset A################################
dataset_A=pd.read_csv("winequality-red.csv",sep=';')

m=len(dataset_A)
for i in range(m):
    if dataset_A.at[i,"quality"]<=6:
        dataset_A.at[i,"quality"]=0
    else :
        dataset_A.at[i,"quality"]=1
   
for i in dataset_A:
    if i == "quality":
        continue    
    mi=min(dataset_A[i])
    mx=max(dataset_A[i])
    for j in range(m):
        dataset_A.at[j,i]=(dataset_A.at[j,i]-mi)/(mx-mi)
dataset_A.to_csv("dataset_A.csv",index=False)   
#####################Dataset A################################


#####################Dataset B################################
dataset_B=pd.read_csv("winequality-red.csv",sep=';')
      
m=len(dataset_B)
for i in range(m):
    if dataset_B.at[i,"quality"]<5:
        dataset_B.at[i,"quality"]=0
    elif dataset_B.at[i,"quality"] == 5 or dataset_B.at[i,"quality"] == 6 :
        dataset_B.at[i,"quality"]=1
    else :
        dataset_B.at[i,"quality"]=2

   
for i in dataset_B:
    if i == "quality":
        continue    
    mu=dataset_B[i].mean()
    sigma=dataset_B[i].std()
    for j in range(m):
        dataset_B.at[j,i]=(dataset_B.at[j,i]-mu)/(sigma)

    mi=min(dataset_B[i])
    mx=max(dataset_B[i])
    step=(mx-mi)/4
    for j in range(m):
        if dataset_B.at[j,i] <= mi+step:
            dataset_B.at[j,i]=0
        elif dataset_B.at[j,i] <= mi+2*step:
            dataset_B.at[j,i]=1
        elif dataset_B.at[j,i] <= mi+3*step:
            dataset_B.at[j,i]=2
        else :
            dataset_B.at[j,i]=3       
            
dataset_B.to_csv("dataset_B.csv",index=False)   

#####################Dataset B################################