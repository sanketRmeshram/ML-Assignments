
import numpy as np 


######################Modules#######################



class Neural_Network():  #This is my object oriented implementation of Neural Network

	def __init__(self):	
		self.batchs=[]
		self.weight=[]
		self.neurons=[]
		self.delta=[]
		self.bias=[]
		self.len_of_input=0
		self.num_hidden_layer=None
		self.neurons_hidden_layer=None
		self.act_func_hidden_layer =None
		self.neurons_output_layer=None
		self.act_func_output_layer=None
		self.learning_rate=None
		self.epochs=None
		self.cache=[]
		self.derivative_hidden_layer=None
		self.derivative_loss=None
		self.Train_x=None
		self.Train_y=None
		self.Test_x=None
		self.Test_y=None


	def data_loader(self,data,test):   #Load the data into object and Tranform it into the way required 
		
		ind=[]
		x=np.array(data[:,:(len(data[0])-1)]).reshape(-1,len(data[0])-1)
		y=np.array(data[:,len(data[0])-1]).reshape(-1,1)
		now=np.zeros((len(data),3))  # make an hot vector of output
		for i in range(len(data)):      
			now[i][int(y[i])-1]=1
		y=now	
		self.len_of_input=len(x[0])
		for i in range(len(data)):
			if(i%32 == 0):
				ind.append([])
			ind[i//32].append(i)
		for i in ind:
			self.batchs.append((x[i].T,y[i].T))
		self.Train_x=x.T
		self.Train_y=y.T
		data=test
		x=np.array(data[:,:(len(data[0])-1)]).reshape(-1,len(data[0])-1)
		y=np.array(data[:,len(data[0])-1]).reshape(-1,1)	
		now=np.zeros((len(data),3))	
		for i in range(len(data)):
			now[i][int(y[i])-1]=1
		y=now			
		self.Test_x=x.T
		self.Test_y=y.T
			


	def weight_initialiser(self):  # initialize matrices of weight and bias with proper dimention and with random value between -1 to 1

		self.weight.append(np.random.rand(self.neurons_hidden_layer[0],self.len_of_input))
		self.bias.append(np.random.rand(self.neurons_hidden_layer[0],1))
		for i in range(self.num_hidden_layer-1):
			self.weight.append(np.random.rand(self.neurons_hidden_layer[i+1],self.neurons_hidden_layer[i]))
			self.bias.append(np.random.rand(self.neurons_hidden_layer[i+1],1))
		self.weight.append(np.random.rand(self.neurons_output_layer,self.neurons_hidden_layer[-1]))
		self.bias.append(np.random.rand(self.neurons_output_layer,1))
		for i in range(len(self.weight)):
			self.weight[i]=2*self.weight[i]-1
			self.bias[i]=2*self.bias[i]-1


	def forward(self,x):  # This is forward propogation of NN.
		self.cache=[]
		self.neurons=[]
		z=(self.weight[0]@x)+self.bias[0]
		self.cache.append(z)
		self.neurons.append(self.act_func_hidden_layer(z))
		for i in range(1,self.num_hidden_layer):
			z=(self.weight[i]@self.neurons[-1])+self.bias[i]
			self.cache.append(z)			
			self.neurons.append(self.act_func_hidden_layer(z))

		z=(self.weight[self.num_hidden_layer]@self.neurons[-1])+self.bias[self.num_hidden_layer]
		self.cache.append(z)			
		self.neurons.append(self.act_func_output_layer(z))
		
	def backward(self,da,x):  # This is forward propogation of NN.
		m=len(x[0])

		for i in reversed(range(1,self.num_hidden_layer)):
			dz=np.multiply(da,self.derivative_hidden_layer(self.neurons[i]))
			dw=(1/m)*(dz@(self.neurons[i-1].T))
			db=(1/m)*(np.sum(dz,axis=1,keepdims=True))
			da=(self.weight[i].T)@dz
			self.weight[i]=self.weight[i]-self.learning_rate*dw  #update weight
			self.bias[i]=self.bias[i]-self.learning_rate*db     #update bias
		dz=np.multiply(da,self.derivative_hidden_layer(self.neurons[0]))

		dw=(1/m)*(dz@(x.T))
		db=(1/m)*(np.sum(dz,axis=1,keepdims=True))
		da=(self.weight[0].T)@dz
		self.weight[0]=self.weight[0]-self.learning_rate*dw
		self.bias[0]=self.bias[0]-self.learning_rate*db

	def MBSGD(self):  #Mini Batch Stochastic Gradient Descent 
		for x,y in self.batchs:
			self.forward(x)  #start forward propogation

			m=len(x[0])
			dz=self.neurons[-1]-y   
			dw=(1/m)*(dz@(self.neurons[self.num_hidden_layer-1].T))
			db=(1/m)*(np.sum(dz,axis=1,keepdims=True))
			da=(self.weight[-1].T)@dz
			self.weight[-1]=self.weight[-1]-self.learning_rate*dw
			self.bias[-1]=self.bias[-1]-self.learning_rate*db

			self.backward(da,x)  #start Backpropogation

	def Train(self,dia):  # This function is Train the model in the way mentioned in assignment

		Accuracy_Train=[]  #This will maintain Accuracy Train after every 10 epochs
		Accuracy_Test=[]   #This will maintain Accuracy Test after every 10 epochs
		Index=[]           #This will main indices 10,20,  ... 200   Specifically for plotting 
		for i in range(1,self.epochs+1):
			self.MBSGD()
			if i%10 == 0:
				Index.append(i)
				Accuracy_Train.append((np.sum(np.multiply(self.predict(self.Train_x),self.Train_y))/len(self.Train_x[0]))*100)
				Accuracy_Test.append((np.sum(np.multiply(self.predict(self.Test_x),self.Test_y))/len(self.Test_x[0]))*100)

		print("Final train accuracy : ",Accuracy_Train[-1])
		print("Final test accuracy : ",Accuracy_Test[-1])	



		import matplotlib.pyplot as plt #plot Training Accuracy and Testing Accuracy in one plot

		plt.plot(Index,Accuracy_Train,color='g',label="Training Accuracy")
		plt.plot(Index,Accuracy_Test,color='b',label="Testing Accuracy")

		plt.ylabel('Accuracy')
		plt.xlabel('Epochs')	
		plt.legend()
		plt.title(dia)
		plt.show()
		# plt.savefig(dia+".png")
		plt.close()



	def predict(self,x):   #predict probablity of classification and return hot vector of prediction
		self.forward(x)
		b=np.zeros_like(self.neurons[-1])
		b[np.argmax(self.neurons[-1],axis=0),range(len(b[0]))]=1
		return b

######################Modules#######################



train=np.load('train.npy') # load Train data
test=np.load('test.npy')  # load Test data


################### Part 1A##################################

print("Part 1A :")
NN1=Neural_Network()
NN1.neurons_hidden_layer=[32]
NN1.neurons_output_layer=3
NN1.num_hidden_layer=1
def sigmoid(x):
	# x=np.clip(x,-500,500)
	return 1/(1+np.exp(-x))
	

NN1.act_func_hidden_layer=sigmoid
def derivative_sigmoid(x):
	# x=np.clip(x,-500,500)
	return np.multiply(x,1-x)
	
NN1.derivative_hidden_layer=derivative_sigmoid
NN1.neurons_output_layer=3
def Softmax(x):
	# x=np.clip(x,-500,500)
	return np.exp(x)/np.sum(np.exp(x),axis=0)
NN1.act_func_output_layer=Softmax 

NN1.learning_rate=.01
NN1.epochs=200

NN1.data_loader(train,test)
NN1.weight_initialiser()
NN1.Train("Accuracy part1A")

################### Part 1A##################################


################### Part 1B##################################

print("Part 1B :")

NN2=Neural_Network()

NN2.neurons_hidden_layer=[64,32]
NN2.neurons_output_layer=3
NN2.num_hidden_layer=2
def ReLU(x):
	return np.maximum(0,x)
	

NN2.act_func_hidden_layer=ReLU
def derivative_ReLU(x):
	y=np.copy(x)
	y[y<=0]=0
	y[y>0]=1
	return y
	
NN2.derivative_hidden_layer=derivative_ReLU
NN2.neurons_output_layer=3

NN2.act_func_output_layer=Softmax 

NN2.learning_rate=.01
NN2.epochs=200

NN2.data_loader(train,test)
NN2.weight_initialiser()
NN2.Train("Accuracy part1B")

################### Part 1B##################################