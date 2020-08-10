#use only pandas
import pandas as pd
import math
class Tree:
	def __init__(self):
		self.is_leaf=False
		self.child={}
		self.label=None
		self.attribute=None
		
def all_same(y):
	return len(list(set(y)))==1
def most_common(y):
	ans={}
	for i in y:
		if i in ans:
			ans[i]+=1
		else :
			ans[i]=1	
	ret=[-1,-1]	
	for val,occ in ans.items()	:
		ret=max(ret,[occ,val])
	return ret[1]	


def entropy(freq):
	ans=0
	tot=0
	for i in freq.values():
		tot+=i 
	for i in freq.values():
		if(i):
			ans-=(i/tot)*math.log2(i/tot)
	return ans
def get_Gain(x,y,attribute):
	freq={}
	for i in y:
		if i in freq:
			freq[i]+=1
		else:
			freq[i]=1	
	ans=entropy(freq)	
	for val in range(4):
		freq={}
		cnt=0
		for i in range(len(y)):
			if x[attribute][i]==val:
				if y[i] in freq:
					freq[y[i]]+=1
				else :
					freq[y[i]]=1	
				cnt+=1
		ans-=(cnt/len(y))*entropy(freq)		
	return ans	
	
def get_best_attribute(x,y,attributes):
	ans=[]
	for atr in attributes:
		ans.append([get_Gain(x,y,atr),atr])
	ans=sorted(ans)
	return ans[-1][1]	

				
def ID3(x,y,attributes):
	root=Tree()
	if all_same(y):
		root.is_leaf=True
		root.label=y[0]
		return root

	if len(attributes)==0 or len(y)<10:
		root.is_leaf=True
		root.label=most_common(y)
		return root

	A=get_best_attribute(x,y,attributes)
	root.attribute=A

	for val in range(4):
		x_new=x.copy()
		y_new=y.copy()
		faltu=[]
		for i in range(len(y)):
			if x_new[A][i]!=val :
				faltu.append(i)
		x_new.drop(x_new.index[faltu],inplace=True)
		x_new.reset_index(inplace=True,drop=True)
		y_new.drop(y_new.index[faltu],inplace=True)
		y_new=y_new.reset_index(drop=True)
		if len(y_new) == 0 :
			child=Tree()
			child.is_leaf=True
			child.label=most_common(y)
			root.child[val]=child
		else:
			root.child[val]=ID3(x_new,y_new,attributes-{A})
	return root		

def predict(node,x):
	if node.is_leaf :
		return node.label
	return predict(node.child[x[node.attribute]],x)	
			
def get_result(node,x):
	y=[]
	for i in range(len(x)):
		y.append(predict(node,x.iloc[i]))
	return pd.DataFrame(y)
	pass

#########################decition Tree############################


#########################Read Data################################

from sklearn.model_selection import KFold


x_total=pd.read_csv("dataset_B.csv")
y_total=x_total["quality"]
del x_total["quality"]

kf=KFold(n_splits=3)
kf.get_n_splits(x_total)

#########################Read Data################################

###################### split data ########################################



###################### split data ########################################

result=open("result3.txt",'w')

###################### my Decition Tree########################################


my_accuracy=0
my_precision=0
my_recall=0

sklearn_accuracy=0
sklearn_precision=0
sklearn_recall=0

for train_index, test_index in kf.split(x_total):
	x_train,x_test=x_total.iloc[train_index].copy(),x_total.iloc[test_index].copy()
	y_train,y_test=y_total.iloc[train_index].copy(),y_total.iloc[test_index].copy()	
	x_train.reset_index(inplace=True,drop=True)
	y_train=y_train.reset_index(drop=True)
	x_test.reset_index(inplace=True,drop=True)
	y_test=y_test.reset_index(drop=True)
	attributes=[]
	for i in x_train:
		attributes.append(i)
	node=ID3(x_train,y_train,set(attributes))
	pridicted_from_my_Dection_tree=get_result(node,x_test)

	from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
	my_accuracy+=accuracy_score(y_test, pridicted_from_my_Dection_tree)
	my_precision+=precision_score(y_test, pridicted_from_my_Dection_tree,average = "macro")
	my_recall+=recall_score(y_test, pridicted_from_my_Dection_tree ,average = "macro")


	###################### my Decition Tree########################################


	####################sklearn######################################
	from sklearn.tree import DecisionTreeClassifier
	clf = DecisionTreeClassifier(criterion="entropy",min_samples_split=10)
	clf=clf.fit(x_train,y_train)
	predicted_from_sklearn=clf.predict(x_test)

	sklearn_accuracy+=accuracy_score(y_test, predicted_from_sklearn)
	sklearn_precision+=precision_score(y_test, predicted_from_sklearn,average = "macro")
	sklearn_recall+=recall_score(y_test, predicted_from_sklearn ,average = "macro")


	####################sklearn######################################


my_accuracy/=3
my_precision/=3
my_recall/=3

sklearn_accuracy/=3
sklearn_precision/=3
sklearn_recall/=3


print("Mean accuracy for my Dicition Tree is " ,my_accuracy,file=result)
print("precision for  my Dicition Tree is ",my_precision,file=result)
print("Recall for my Dicition Tree is ",my_recall,file=result)

print("Mean accuracy for sklearn is " ,sklearn_accuracy,file=result)
print("precision for  sklearn is ",sklearn_precision,file=result)
print("Recall for sklearn is ",sklearn_recall,file=result)