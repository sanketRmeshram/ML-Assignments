#take input from .csv file
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sys

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

def PolyCoefficients(coeffs,x):
    X=[x**i for i in range(len(coeffs))]
    ans=0
    for i in range(len(coeffs)):
        ans+=X[i]*coeffs[i]
    return ans    
    pass

cofficient=open("Lasso.txt",'r')
lamb=[.25,.5,.75,1]
for i in lamb:
	
	x=np.linspace(-.1,1.1,1000)    
	plt.scatter(train.iloc[:,0],train.iloc[:,1],color='g',s=1,label="Training Data")
	plt.scatter(test.iloc[:,0],test.iloc[:,1],color='r',s=1,label="Training Data")
	
	
	plt.ylim(-1.1,1.1)
	plt.xlabel('Label')
	plt.ylabel('Feature')	
	plt.title("lasso Curve for lambda = " + str(i))
	
	poly=[float(j) for j in cofficient.readline().split()]
	plt.plot(x,PolyCoefficients(poly,x),color='b',label="deg 1")
	poly=[float(j) for j in cofficient.readline().split()]
	plt.plot(x,PolyCoefficients(poly,x),color='y',label="deg 9")	
	plt.legend()
	plt.savefig("lasso for lambda "+str(i) +".png")
	plt.close()

cofficient=open("Ridge.txt",'r')
lamb=[.25,.5,.75,1]
for i in lamb:
	
	x=np.linspace(-.1,1.1,1000)    
	plt.scatter(train.iloc[:,0],train.iloc[:,1],color='g',s=1,label="Training Data")
	plt.scatter(test.iloc[:,0],test.iloc[:,1],color='r',s=1,label="Training Data")
	
	
	plt.ylim(-1.1,1.1)
	plt.xlabel('Label')
	plt.ylabel('Feature')	
	plt.title("ridge Curve for lambda = " + str(i))
	
	poly=[float(j) for j in cofficient.readline().split()]
	plt.plot(x,PolyCoefficients(poly,x),color='b',label="deg 1")
	poly=[float(j) for j in cofficient.readline().split()]
	plt.plot(x,PolyCoefficients(poly,x),color='y',label="deg 9")	
	plt.legend()
	plt.savefig("ridge for lambda "+str(i) +".png")
	plt.close()

err=open("Result_lasso.txt",'r')
err1=open("Result_ridge.txt",'r')
lamb=[.25,.5,.75,1]
for i in lamb:
	error_Train=[]
	error_Test=[]
	line=err.readline().split()
	error_Train.append(float(line[len(line)-1]))
	line=err.readline().split()
	error_Test.append(float(line[len(line)-1]))		
	line=err.readline().split()
	error_Train.append(float(line[len(line)-1]))
	line=err.readline().split()
	error_Test.append(float(line[len(line)-1]))		
	N=[1,9]	
	plt.scatter(N,error_Train,color='g',s=5,label="Training Error lasso")
	plt.scatter(N,error_Test,color='r',s=5,label="Testing Error lasso")	
	plt.ylabel('Error')
	plt.xlabel('Degree')	
	
	plt.title( "Degree vs Error lambda="+str(i))
	

	error_Train=[]
	error_Test=[]
	line=err1.readline().split()
	error_Train.append(float(line[len(line)-1]))
	line=err1.readline().split()
	error_Test.append(float(line[len(line)-1]))		
	line=err1.readline().split()
	error_Train.append(float(line[len(line)-1]))
	line=err1.readline().split()
	error_Test.append(float(line[len(line)-1]))		
	plt.scatter(N,error_Train,color='b',label="Training Error ridge",marker='s',s=5)
	plt.scatter(N,error_Test,color='y',label="Testing Error ridge",marker='s',s=5)		
	plt.legend()
	plt.savefig("Degree vs Error lambda="+str(i)+" .png")
	plt.close()	


