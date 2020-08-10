
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

cofficient=open("Cofficient.txt",'r')
coff=[]
for i in range(9):
	coff.append([float(j) for j in cofficient.readline().split()])

for poly in coff :
	x=np.linspace(-.1,1.1,1000)    
	plt.scatter(train.iloc[:,0],train.iloc[:,1],color='g',s=1,label="Training Data")
	plt.scatter(test.iloc[:,0],test.iloc[:,1],color='r',s=1,label="Training Data")
	plt.plot(x,PolyCoefficients(poly,x),color='b',label="Curve")
	
	plt.legend()
	plt.ylim(-1.1,1.1)
	plt.xlabel('Label')
	plt.ylabel('Feature')	
	plt.title("Curve of Degree : " + str(len(poly)-1))
	plt.savefig("deg "+str((len(poly)-1)) +".png")
	plt.close()

err=open("Result.txt",'r')
error_Test=[]
error_Train=[]
for i in range(9):
	line=err.readline().split()
	error_Train.append(float(line[len(line)-1]))
	line=err.readline().split()
	error_Test.append(float(line[len(line)-1]))	

N=[i for i in range(1,10)]
plt.scatter(N,error_Train,color='g',s=6,label="Training Error")
plt.scatter(N,error_Test,color='r',s=6,label="Testing Error")
plt.legend()
plt.xlabel('Error')
plt.ylabel('Degree')	
plt.title( "  Degree vs Error ")
plt.savefig("Degree vs Error.png")
plt.close()