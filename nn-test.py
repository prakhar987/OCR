import math
import numpy as np
from random import randint

def downsize(dataset):
    a=[]
    for i in range(len(dataset)):
        tmp=[]
        j=0
        while(j<32):
            k=0
            while (k<32):
                sm=0
                for l in range(j,j+4):
                    for m in range(k,k+4):
                        sm=sm+int(dataset[i][l*32+m])
                if(sm>8):
                    tmp.append(1)
                else:
                    tmp.append(0)
                k+=4
            j+=4
            a.append(tmp)
    
    return a

def read_data(name_file):
    f=open(name_file, "r")
    a=[]
    for line in f:

        if(len(line)==33):
            a.append(line[:32]) # trim newline

        if(len(line)==3):
            ## Store digit value
            digit=str(line)
            digit=digit.strip()
            if(digit=='0' or digit=='1' or digit=='3'):
                true_value.append(digit)
                ## Now store data set
                b=[]
                for i in range (len(a)):
                    for j in range(len(a[i])):
                        b.append(a[i][j])
                dataset.append(b) 
                # break
            ## Reset a
            a=[]    
            
    
    # print(dataset)
    # print(true_value)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


def build_model(wj,wk,nH,total_reps):
    eta=0.5
    for i in range(total_reps):

        index=randint(0,len(data)-1)
        xi = data[index]
        netj = np.dot(xi,wj) 
        yj=sigmoid(netj)
        netk=np.dot(yj,wk)
        zk = sigmoid(netk)

        ## Now do backprop
        #First Layer
        # error=true_val[index] - zk
        # delta_k=error*derivative(netk)
        # tmp=eta*delta_k
        # p = np.zeros(shape=(1,nH))
        # for k in range(nH):
        #     p[0][k]=yj[k]
        # q = np.zeros(shape=(1,2))
        # q[0][0]=tmp[0]
        # q[0][1]=tmp[1]
        # tmp=p.T.dot(q)
        # copywk=wk[:]
        # wk = wk + tmp

        # tk = mapping(numbers[randomIndex])
        copywk = wk[:]
        deltak = []
        for k in range(2):
            error = true_val[index][k] - zk[k]
            dell = error * derivative(netk[k])
            deltak.append(dell)
            for j in range(nH):
                delta = eta * dell * yj[j]
                wk[j][k] = wk[j][k] + delta
        ##Second layer

        deltaj = []
        for j in range(nH):
            sum = 0
            for k in range(2):
                sum += (copywk[j][k] * deltak[k])
                # print sum
            dell = sum * derivative(netj[j])
            # print dell
            deltaj.append(dell)
            for i in range(64):
                delta = eta * dell * xi[i]
                wj[i][j] = wj[i][j] + delta
         
    return(wk,wj)


def feed_forward(wj,wk,total_reps):
    counter=0
    for i in range(total_reps):

        index=randint(0,len(data)-1)
        xi = data[index]
        netj = np.dot(xi,wj) 
        yj=sigmoid(netj)
        netk=np.dot(yj,wk)
        zk = sigmoid(netk)
        # print("True Value=",true_val[index],"Output=",zk)
        if(zk[0]>0.5):
            zk[0]=1
        else:
            zk[0]=0
        if(zk[1]>0.5):
            zk[1]=1
        else:
            zk[1]=0

        if(zk[0]==true_val[index][0] and zk[1]==true_val[index][1]):
            counter+=1
    print("Correct Answers in % =",(counter/200)*100)

        
        
        

dataset=[] ## Stores the datasets
true_value=[]## Stores true values 

read_data("optdigits-orig.tra")
# print(len(dataset[0]))
dataset=downsize(dataset)

## Now we have the lists , copy to numpy lists
data = np.zeros(shape=(len(dataset),64))
for i in range(len(dataset)):
    for j in range(64):
        data[i][j]=dataset[i][j]
# print(type(data))
true_val = np.zeros(shape=(len(dataset),2))
for i in range (len(true_value)):
    if(true_value[i]=='0'):
        true_val[i][0]=0
        true_val[i][1]=0
    if(true_value[i]=='1'):
        true_val[i][0]=0
        true_val[i][1]=1
    if(true_value[i]=='3'):
        true_val[i][0]=1
        true_val[i][1]=0


#### Now begin

# init to random values
np.random.seed(1)
nH=20
# randomly initialize our weights with mean 0
wj = 2*np.random.random((64,nH)) - 1
wk = 2*np.random.random((nH,2)) - 1
wk,wj=build_model(wj,wk,nH,5000)
feed_forward(wj,wk,100)

## Printing Weights : 
for i in range(len(wj)):
	print(wj[i])
print("NOW FINAL LAyER")
print(wk)

### Now process CV file
dataset=[] ## Stores the datasets
true_value=[]## Stores true values 

read_data("optdigits-orig.cv")
# print(len(dataset[0]))
dataset=downsize(dataset)

## Now we have the lists , copy to numpy lists
data = np.zeros(shape=(len(dataset),64))
for i in range(len(dataset)):
    for j in range(64):
        data[i][j]=dataset[i][j]
# print(type(data))
true_val = np.zeros(shape=(len(dataset),2))
for i in range (len(true_value)):
    if(true_value[i]=='0'):
        true_val[i][0]=0
        true_val[i][1]=0
    if(true_value[i]=='1'):
        true_val[i][0]=0
        true_val[i][1]=1
    if(true_value[i]=='3'):
        true_val[i][0]=1
        true_val[i][1]=0

feed_forward(wj,wk,200)





