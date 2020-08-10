from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
x=pd.read_csv("AllBooks_baseline_DTM_Labelled_modified.csv")
del x["Unnamed: 0"]
x=np.array(x)
pca=PCA(n_components=100)
x=pca.fit_transform(x)
x=np.array(x)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tf_idf_vector=tfidf_transformer.fit_transform(x)
x=tf_idf_vector.toarray()

#########################################aggomerative###############################
class cluster:
	clus={}
	def __init__(self, arg=[]):
		self.clus=set(arg)

def  dist_aggo(a,b,pre):  # min similarity = maximum distance
	return min([pre[i][j] for i in a.clus for j in b.clus])

def get_min_pair(x,A,pre):
	g1,g2=-1,-1
	d=-1
	temp=list(A)
	for i in range(len(temp)):
		for j in range(i):
			if(i!=j):
				now=dist_aggo(temp[i],temp[j],pre)
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
			pre[i][j]=pre[j][i]=np.exp(-(x[i].dot(x[j].T)))			

	A=set(A)
	while len(A)>8:
		g1,g2=get_min_pair(x,A,pre)
		A=A-{g1}
		A=A-{g2}
		A.add(union(g1,g2))		
	return A	

#########################################aggomerative###############################

#########################################Kmeans###############################

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
#########################################Kmeans###############################

A=HAC(x)
result=open("agglomerative_reduced.txt",'w')
A=[sorted(list(i.clus)) for i in A]
A=sorted(A)	
for i in A:
	print(*sorted(list(i)), sep=",",file = result)

A=K_means(x,8)
result=open("kmeans_reduced.txt",'w')
temp=[[] for i in range(8)]
for i in range(len(A)):
	temp[A[i]].append(i)
for i in range(len(temp)):
	temp[i]=sorted(list(temp[i]))
temp=sorted(temp)		
for i in temp:
	print(*sorted(i),sep=",", file = result)
