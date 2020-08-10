import numpy as np 
from sklearn.neural_network import MLPClassifier

def Accuracy(y,target):  #Returns accuracy

	correct=0
	for i in range(len(y)):
		if(y[i]==target[i]):
			correct+=1
	return (correct/len(y))*100		


Train=np.load('train.npy') # load Train data
Test=np.load('test.npy')  # load Test data
Train_x=np.array(Train[:,:(len(Train[0])-1)]).reshape(-1,len(Train[0])-1)  #Split data into x and y
Train_y=np.array(Train[:,len(Train[0])-1])

Test_x=np.array(Test[:,:(len(Test[0])-1)]).reshape(-1,len(Test[0])-1)  #Split data into x and y
Test_y=np.array(Test[:,len(Test[0])-1])


print("Part 2 Specification 1A :")
#MLPClassifier with specification in part 1A
NN1=MLPClassifier(solver='sgd',hidden_layer_sizes=(32),activation='logistic',learning_rate='constant',learning_rate_init=.01,batch_size=32,max_iter=200)
NN1.fit(Train_x,Train_y)
print("Final train accuracy : ",Accuracy(NN1.predict(Train_x),Train_y))
print("Final test accuracy : ",Accuracy(NN1.predict(Test_x),Test_y))

print("Part 2 Specification 1B :")
#MLPClassifier with specification in part 1A
NN2=MLPClassifier(solver='sgd',hidden_layer_sizes=(64,32),activation='relu',learning_rate='constant',learning_rate_init=.01,batch_size=32,max_iter=200)
NN2.fit(Train_x,Train_y)
print("Final train accuracy : ",Accuracy(NN2.predict(Train_x),Train_y))
print("Final test accuracy : ",Accuracy(NN2.predict(Test_x),Test_y))

