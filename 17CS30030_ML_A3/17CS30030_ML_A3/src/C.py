import pandas as pd 
import numpy as np 


def dist(a,b):
	return np.exp(-((a.T).dot(b)))

def assign_cluster(x,centroid):
	ret=[-1 for i in range(len(x))]
	for i in range(len(x)):
		d=[dist(x[i].reshape(-1,1),j.reshape(-1,1)) for j in centroid]
		ans=d[0]
		ind=0
		for j in range(1,8):
			if(d[j]<ans):
				ans=d[j]
				ind=j
		ret[i]=ind
	return ret

def squared_error(x,ass,centroid):
	ans=0
	for i in range(len(x)):
		ans+=(np.sum(pow(x[i].reshape(-1,1)-centroid[ass[i]].reshape(-1,1),2)))
	return ans	

def new_centroid(x,ass):
	temp=[[] for i in range(8)]
	for i in range(len(ass)):
		temp[ass[i]].append(i)
	ans=[]	
	for i in temp:
		now=np.zeros(x[0].shape[0]).reshape(-1,1)
		for j in i:
			now+=x[j].reshape(-1,1)
		now/=len(i)
		ans.append(now)
	return ans		

def chk(centroid,prev):  # this checks if centroid have changed or not 
	if(len(centroid)!=len(prev)):
		return False 
	for i in range(len(centroid)):
		if not np.array_equal(centroid[i],prev[i]):
			return False
	return True		

def K_means(x,k):
	index=[i for i in range(len(x))]
	import random
	random.shuffle(index)
	centroid=[x[index[i]].reshape(-1,1) for i in range(8)]   #select random cluster
	ass=assign_cluster(x,centroid)
	prev=[]
	while not chk(centroid,prev):
		ass=assign_cluster(x,centroid)
		prev=centroid
		centroid=new_centroid(x,ass)
	return assign_cluster(x,centroid)

x=np.load('mat.npy')
A=K_means(x,8)
result=open("kmeans.txt",'w')
temp=[[] for i in range(8)]
for i in range(len(A)):
	temp[A[i]].append(i)

for i in range(len(temp)):
	temp[i]=sorted(list(temp[i]))
temp=sorted(temp)		
for i in temp:
	print(*sorted(i), sep=",",file = result)
