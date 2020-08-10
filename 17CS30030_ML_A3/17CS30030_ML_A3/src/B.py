import pandas as pd 
import numpy as np 

class cluster:
	clus={}
	def __init__(self, arg=[]):
		self.clus=set(arg)

def  dist(a,b,pre):  # min similarity = maximum distance  
	return min([pre[i][j] for i in a.clus for j in b.clus])

def get_min_pair(x,A,pre):
	g1,g2=-1,-1
	d=-1
	temp=list(A)
	for i in range(len(temp)):
		for j in range(i):
			if(i!=j):
				now=dist(temp[i],temp[j],pre)
				if d==-1 or d > now:
					g1,g2=i,j
					d=now		
	return temp[g1],temp[g2]				
def union(a,b):
	temp=cluster()
	temp.clus=a.clus | b.clus
	return temp

def HAC(x):
	A=([cluster([i])  for i in range(len(x))])
	pre=np.zeros((len(x),len(x)))
	for i in range(len(x)):  
		for j in range(i):
			pre[i][j]=pre[j][i]=np.exp(-(x[i].dot(x[j].T))) # precompute distance between each pair this will reduce execution time a lot			

	A=set(A)
	while len(A)>8:
		g1,g2=get_min_pair(x,A,pre)
		A=A-{g1}
		A=A-{g2}
		A.add(union(g1,g2))		
	return A	

from numpy import genfromtxt

x=np.load('mat.npy')
A=HAC(x)
result=open("agglomerative.txt",'w')
A=[sorted(list(i.clus)) for i in A]
A=sorted(A)	
for i in A:
	print(*sorted(list(i)), sep=",",file = result)


