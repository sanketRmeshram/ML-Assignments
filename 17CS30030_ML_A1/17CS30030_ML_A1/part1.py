#take input from .csv file
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sys
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

plt.scatter(train.iloc[:,0],train.iloc[:,1],color='g',s=1,label="Training Data")
plt.scatter(test.iloc[:,0],test.iloc[:,1],color='r',s=1,label="Training Data")
plt.xlabel('Label')
plt.ylabel('Feature')
plt.legend()
plt.savefig('data.png')
plt.close()


def func(W,X):        
    ans=W.transpose().dot(X)
    ans=np.sum(ans)
    return ans 

def cost(X,Y,W):
    ans=(W.transpose()).dot(X)
    ans=ans.transpose()
    ans=np.sum(pow(ans-Y,2))
    return ans/(2*len(Y))   


def move_one_step(X,Y,alpha,W):
    out=W.transpose().dot(X)
    out=out.transpose()
    out=out-Y
    out=X.dot(out)
    out=W-((alpha/len(Y))*out)
    return out

def graient_decent(X,Y,alpha,W):
    prev=W
    W=move_one_step(X,Y,alpha,W)
    while cost(X,Y,prev)-(cost(X,Y,W)) > .00000001:
        prev=W
        W=move_one_step(X,Y,alpha,W)

    return W

alpha=.05
result=open("Result.txt",'w')
cofficient=open("Cofficient.txt",'w')

for i in range(1,10):
    X=train.iloc[:,0]
    X=[ [x**deg for deg in range(i+1)] for x in X ]
    X=np.array(X)
    X=X.transpose()
    Y=train.iloc[:,1]     
    Y=np.array(Y).reshape(-1,1)
    W=np.zeros((i+1,1))
    W=graient_decent(X,Y,alpha,W)
    for coff in W:
        print(np.sum(coff),sep=' ',file=cofficient,end=' ')
    print(file=cofficient)    

    print("squared error on train data when deg = ",i," is : ",cost(X,Y,W),file=result)

    X=test.iloc[:,0]
    X=[ [x**deg for deg in range(i+1)] for x in X ]
    X=np.array(X)
    X=X.transpose()
    Y=test.iloc[:,1]     
    Y=np.array(Y).reshape(-1,1)

    print("squared error on test data when deg = ",i," is : ",cost(X,Y,W),file=result)
