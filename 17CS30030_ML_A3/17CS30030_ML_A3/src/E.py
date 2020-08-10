from math import log2

def NMI(cluster,x):
	freq={}
	for i in range(len(x)):
		now=str(x.iloc[i,0])
		if now in freq:
			freq[now].append(i)
		else:
			freq[now]=[i]
	entrophy_of_class=0
	entrophy_of_cluster=0
	for i in cluster:
		if len(i):
			entrophy_of_cluster-=(len(i)/len(x))*log2((len(i)/len(x)))
	for i in freq.values():
		if len(i):
			entrophy_of_class-=(len(i)/len(x))*log2((len(i)/len(x)))			
	mutual_information=entrophy_of_class
	for i in cluster:
		ans=0
		freq={}
		for j in i:
			now=str(x.iloc[j,0])
			if now in freq:
				freq[now].append(j)
			else:
				freq[now]=[j]
		for j in freq.values():
			ans-=(len(j)/len(i))*log2((len(j)/len(i)))
		mutual_information-=ans*(len(i)/len(x))
	return 	(2*mutual_information)/(entrophy_of_class+entrophy_of_cluster)

import pandas as pd 
x=pd.read_csv("AllBooks_baseline_DTM_Labelled_modified.csv")

result=open("agglomerative.txt",'r')
cluster=[]
for i in range(8):
	cluster.append([int(j)  for j in result.readline().split(',') ])
print("NMI for agglomerative is : ",NMI(cluster,x))

result=open("agglomerative_reduced.txt",'r')
cluster=[]
for i in range(8):
	cluster.append([int(j)  for j in result.readline().split(',') ])
print("NMI for agglomerative_reduced is : ",NMI(cluster,x))

result=open("kmeans.txt",'r')
cluster=[]
for i in range(8):
	cluster.append([int(j)  for j in result.readline().split(',') ])
print("NMI for kmeans is : ",NMI(cluster,x))

result=open("kmeans_reduced.txt",'r')
cluster=[]
for i in range(8):
	cluster.append([int(j)  for j in result.readline().split(',') ])
print("NMI for kmeans_reduced is : ",NMI(cluster,x))