import numpy as np 
def PreProcess(data):   
	for j in range(len(data[0])-1):  # This loop will normalize the data
		mean=0
		sq_mean=0
		for i in range(len(data)):
			mean+=data[i][j]
			sq_mean+=data[i][j]*data[i][j]
		mean/=len(data)
		sq_mean/=len(data)
		standered_deviation=(sq_mean-mean*mean)**.5
		for i in range(len(data)):
			data[i][j]=(data[i][j]-mean)/standered_deviation
	np.random.shuffle(data)  #now we will divide the train , test to 80% and 20% respectively 
	sz=int(.8*len(data))
	return data[:sz],data[sz:]



##############PreProcess###################

file=open("seeds_dataset.txt","r")

data=[]

while True:           #This loop will take input from file
	now=file.readline()
	if(len(now)==0):
		break
	data.append([float(i) for i in now.split()])

data=np.array(data)

train,test=PreProcess(data)
np.save('train.npy',train)  #Save train data
np.save('test.npy',test)   #Save test data

##############PreProcess###################