import numpy as np 
import pandas as pd 
import math 

################logistic reg##############################
def func(x,theta):
    return 1/(1+np.exp(-(x.dot(theta))))
     
def cost(x,y,theta):
    ans=func(x,theta)
    for i in range(len(y)):
        if y[i]==0 :
            ans[i]=1-ans[i]
    ans=np.log(ans)
    ans=np.sum(ans)
    return -(ans/len(y))
def move_one_step(x,y,theta,alpha):
    ans=func(x,theta)
    ans=ans-y
    ans=theta-(alpha/len(y))*(x.transpose().dot(ans))
    return ans

def gradient_decent(x,y,theta,alpha):
    prev=theta
    theta=move_one_step(x,y,theta,alpha)
    while abs(cost(x,y,prev)-cost(x,y,theta)) > .000005 :
        prev=theta
        theta=move_one_step(x,y,theta,alpha)
    return theta

################logistic reg##############################


x_total=pd.read_csv("dataset_A.csv")
y_total=x_total["quality"]
del x_total["quality"]



###################### split data ########################################

from sklearn.model_selection import KFold

kf=KFold(n_splits=3,shuffle=True,random_state=5)
kf.get_n_splits(x_total)

###################### split data ########################################

result=open("result2.txt",'w')

######################my log reg########################################

my_accuracy=0
my_precision=0
my_recall=0

sklearn_accuracy=0
sklearn_precision=0
sklearn_recall=0

for train_index, test_index in kf.split(x_total):
	x_train,x_test=x_total.iloc[train_index],x_total.iloc[test_index]
	y_train,y_test=y_total.iloc[train_index],y_total.iloc[test_index]

	x=x_train.copy()
	y=y_train.copy()
	x.insert(0,"zero condition",[1 for i in range(len(x))])
	x=np.array(x)
	y=np.array(y).reshape(-1,1)
	theta=[0 for i in range(len(x[0]))]
	theta=np.array(theta).reshape(-1,1)
	theta=gradient_decent(x,y,theta,.2)
	x=x_test.copy()
	x.insert(0,"zero condition",[1 for i in range(len(x))])
	x=np.array(x)
	pridicted_from_my_log_reg=func(x,theta)
	for i in range(len(pridicted_from_my_log_reg)):
	    if(pridicted_from_my_log_reg[i]>=.5):
	        pridicted_from_my_log_reg[i]=1
	    else :
	        pridicted_from_my_log_reg[i]=0    

	from sklearn.metrics import  accuracy_score, precision_score, recall_score
	my_accuracy+=accuracy_score(y_test, pridicted_from_my_log_reg)
	my_precision+=precision_score(y_test, pridicted_from_my_log_reg)
	my_recall+=recall_score(y_test, pridicted_from_my_log_reg )

######################my log reg########################################


######################sklearn########################################

	from sklearn.linear_model import LogisticRegression
	classifier = LogisticRegression(penalty='none',solver='saga')
	classifier.fit(x_train,y_train)
	pridicted_from_sklearn=classifier.predict(x_test)

	sklearn_accuracy+=accuracy_score(y_test, pridicted_from_sklearn)
	sklearn_precision+=precision_score(y_test, pridicted_from_sklearn)
	sklearn_recall+=recall_score(y_test, pridicted_from_sklearn )



######################sklearn########################################


my_accuracy/=3
my_precision/=3
my_recall/=3

sklearn_accuracy/=3
sklearn_precision/=3
sklearn_recall/=3


print("Mean accuracy for my log reg is " ,my_accuracy,file=result)
print("precision for  my log reg is ",my_precision,file=result)
print("Recall for my log reg is ",my_recall,file=result)

print("Mean accuracy for sklearn is " ,sklearn_accuracy,file=result)
print("precision for  sklearn is ",sklearn_precision,file=result)
print("Recall for sklearn is ",sklearn_recall,file=result)