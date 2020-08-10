
#take input from .csv file
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sys
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

alpha=.05

def cost(X,Y,W):
    ans=(W.transpose()).dot(X)
    ans=ans.transpose()
    ans=np.sum(pow(ans-Y,2))
    return ans/(2*len(Y))   


def func_lasso(W,X):        
    ans=W.transpose().dot(X)
    ans=np.sum(ans)
    return ans 

def cost_lasso(X,Y,W,lamb):
    ans=(W.transpose()).dot(X)
    ans=ans.transpose()
    ans=np.sum(pow(ans-Y,2))
    ans+=lamb*np.sum(W)
    ans-=lamb*W[0]
    ans=ans/(2*len(Y))
    return ans


def move_one_step_lasso(X,Y,alpha,W,lamb):
    out=W.transpose().dot(X)
    out=out.transpose()
    out=out-Y
    out=X.dot(out)
    
    temp=np.array([lamb for i in range(len(W))]).reshape(-1,1)
    temp[0]=0
    out=out+temp
    out=(1/len(Y))*out
    out=W-(alpha*out)
    return out

def graient_decent_lasso(X,Y,alpha,W,lamb):
    prev=W
    W=move_one_step_lasso(X,Y,alpha,W,lamb)
    while cost_lasso(X,Y,prev,lamb)-(cost_lasso(X,Y,W,lamb)) > .0000001:
        prev=W
        W=move_one_step_lasso(X,Y,alpha,W,lamb)
    return W

# max training error is in deg 1 
# min training error is in deg 9
lasso=open("Lasso.txt",'w')
result_lasso=open("Result_lasso.txt",'w')

lamb=[.25,.5,.75,1]
for i in lamb:
    degree=[1,9]

    for j in degree:
        X=train.iloc[:,0]
        X=[ [x**deg for deg in range(j+1)] for x in X ]
        X=np.array(X)
        X=X.transpose()
        Y=train.iloc[:,1]     
        Y=np.array(Y).reshape(-1,1)
        W=np.zeros((j+1,1))

        W=graient_decent_lasso(X,Y,alpha,W,i)


        for coff in W:
            print(np.sum(coff),sep=' ',file=lasso,end=' ')
        print(file=lasso)    
      

        print("squared error on train data when deg = ",j," is : ",cost(X,Y,W),file=result_lasso)

        X=test.iloc[:,0]
        X=[ [x**deg for deg in range(j+1)] for x in X ]
        X=np.array(X)
        X=X.transpose()
        Y=test.iloc[:,1]     
        Y=np.array(Y).reshape(-1,1)

        print("squared error on test data when deg = ",j," is : ",cost(X,Y,W),file=result_lasso)




def func_ridge(W,X):        
    ans=W.transpose().dot(X)
    ans=np.sum(ans)
    return ans 

def cost_ridge(X,Y,W,lamb):
    ans=(W.transpose()).dot(X)
    ans=ans.transpose()
    ans=np.sum(pow(ans-Y,2))
    
    ans+=lamb*np.sum(pow(W,2))
    ans-=lamb*W[0]*W[0]
    ans=ans/(2*len(Y))
    return ans

def move_one_step_ridge(X,Y,alpha,W,lamb):
    out=W.transpose().dot(X)
    out=out.transpose()
    out=out-Y
    out=X.dot(out)
    
    temp=np.array([i for i in W]).reshape(-1,1)
    temp[0]=0
    out=out+lamb*temp
    out=(1/len(Y))*out
    out=W-(alpha*out)
    return out

def graient_decent_ridge(X,Y,alpha,W,lamb):
    prev=W
    W=move_one_step_ridge(X,Y,alpha,W,lamb)
    while cost_ridge(X,Y,prev,lamb)-(cost_ridge(X,Y,W,lamb)) > .0000001:
        prev=W
        W=move_one_step_ridge(X,Y,alpha,W,lamb)
    return W

ridge=open("Ridge.txt",'w')
result_ridge=open("Result_ridge.txt",'w')

lamb=[.25,.5,.75,1]

for i in lamb:
    degree=[1,9]
    for j in degree:
        X=train.iloc[:,0]
        X=[ [x**deg for deg in range(j+1)] for x in X ]
        X=np.array(X)
        X=X.transpose()
        Y=train.iloc[:,1]     
        Y=np.array(Y).reshape(-1,1)
        W=np.zeros((j+1,1))
        W=graient_decent_ridge(X,Y,alpha,W,i)


        for coff in W:
            print(np.sum(coff),sep=' ',file=ridge,end=' ')
        print(file=ridge)    


        print("squared error on train data when deg = ",j," is : ",cost(X,Y,W),file=result_ridge)

        X=test.iloc[:,0]
        X=[ [x**deg for deg in range(j+1)] for x in X ]
        X=np.array(X)
        X=X.transpose()
        Y=test.iloc[:,1]     
        Y=np.array(Y).reshape(-1,1)

        print("squared error on test data when deg = ",j," is : ",cost(X,Y,W),file=result_ridge)

