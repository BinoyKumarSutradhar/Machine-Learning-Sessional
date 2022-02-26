
from sklearn.utils.extmath import softmax
import gzip
from turtle import forward, shape
from matplotlib.pyplot import cla
import numpy as np
import random
import math

image_size = 28
num_images = 5
clr_chnls = 1
sd = 20

f = gzip.open('train-images-idx3-ubyte.gz','r')
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, clr_chnls, image_size, image_size)


f1 = gzip.open('train-labels-idx1-ubyte.gz','r')
f1.read(8)  
buf = f1.read(num_images)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
labels = labels.reshape(num_images,1)


def submatrix(matrix, startRow, startCol, size):
    return matrix[startRow:startRow+size,startCol:startCol+size]

###########################################################################

class convolution:
    def __init__(self, img, input_size, filter_size, filter_cnt, pad, stride):
        random.seed(sd)
        self.filter_cnt = filter_cnt
        self.filter = np.random.randn(filter_cnt, filter_size, filter_size) * 0.1
        self.filter_size = filter_size
        self.img = img
        self.output_size = int(((input_size - filter_size + (2*pad)) / stride) + 1)
        self.output = np.zeros((num_images,filter_cnt, self.output_size, self.output_size))
        self.bias = np.random.randn(filter_cnt, self.output_size, self.output_size) * 0.1
        self.stride = stride
        self.pad = pad
    
    def forward(self):
        
        for p in range(num_images):
            photo = np.pad(self.img[p], pad_width=[(0, 0),(self.pad, self.pad),(self.pad, self.pad)], mode='constant')
            for f in range(self.filter_cnt):

                fil = self.filter[f]
                fil = np.array(fil)
                bs = self.bias[f]
                bs = np.array(bs)
                out = np.zeros(self.output_size * self.output_size)
                for c in range(clr_chnls):
                    v=0

                    inp = photo[c]
                    inp = np.array(inp)
                    temp = np.zeros((self.filter_size, self.filter_size))

                    s= len(temp[0])
                    filter_s = self.filter_size
                    inp_s = len(inp[0])
                    strd = self.stride

                    for r in range(0,inp_s,strd):
                        if r> inp_s-filter_s:
                            break
                        for i in range(0,inp_s,strd):
                            if i>=0 and i<=inp_s-filter_s:
                                temp = submatrix(inp,r,i,filter_s)
                                temp = np.array(temp)
                                s= len(temp[0])
                                sum = 0
                                for j in range(s):
                                    for k in range(s):
                                        sum = sum + (temp[j][k]*fil[j][k])
                                out[v] = out[v]+sum
                            else:
                                break
                out = out.reshape(self.output_size, self.output_size)
                out = out + bs
                self.output[p][f] = out
                
            return self.output


############# Convolution Layer #################### 

##################  RELU  ##########################

class relu:
    def __init__(self, out):
        self.out = out

    def forward(self):
        
        for p in range(num_images):
            for c in range(self.out.shape[1]): # num of filters
                temp = self.out[p][c]
                for i in range(self.out.shape[2]):
                    for j in range(self.out.shape[2]):
                        if temp[i][j] < 0:
                            temp[i][j] = 0
                self.out[p][c] = temp
        return self.out

##################  RELU  ##########################

##################  POOLING  ##########################  

class pool:
    def __init__(self, out,filter_size,stride):
        self.out = out
        self.filter_size = filter_size
        self.stride = stride
        self.output_size = int( ((out.shape[2]-filter_size)/stride)+1 )
        self.output = np.zeros((num_images,out.shape[1], self.output_size, self.output_size))
    
    def forward(self):

        inp_s = self.out.shape[2]
        filter_s = self.filter_size
        strd = self.stride
        for p in range(num_images):
            for c in range(self.out.shape[1]):
                inp = self.out[p][c]
                out = []
                for r in range(0,inp_s,strd):
                    if r> inp_s-filter_s:
                        break
                    for i in range(0,inp_s,strd):
                        if i>=0 and i<=inp_s-filter_s:
                            temp = submatrix(inp,r,i,filter_s)
                            temp = np.array(temp)
                            x = np.amax(temp)
                            out.append(x)
                            
                        else:
                            break
                out = np.array(out)
                out = out.reshape(self.output_size, self.output_size)
                self.output[p][c] = out
        
        return self.output

##################  POOLING  ##########################


############## FULLY CONNECTED LAYER  ######################

class full_connect:
    def __init__(self,  out, inp_s, out_s):
        random.seed(sd)
        self.inp_s = inp_s
        self.weight = np.random.randn(out_s, inp_s) * 0.1
        self.bias = np.random.randn(out_s, 1) * 0.1
        self.out = out
        self.output = np.zeros((num_images,out_s, 1))

    def forward(self):
        for p in range(self.out.shape[0]):
            photo = self.out[p]
            photo = photo.reshape(self.inp_s,1)
            result = np.dot(self.weight, photo) + self.bias
            self.output[p] = result
        
        return self.output


############## FULLY CONNECTED LAYER  ######################

###############  SOFTMAX  #############################

class softmax:
    def __init__(self,  out):
        self.out = out
        self.output = np.zeros((num_images,out.shape[1], 1))
    
    def forward(self):
        for p in range(num_images):
            photo = self.out[p]
            e_x = np.exp(photo - np.max(photo))
            result =  e_x / e_x.sum()
            self.output[p] = result
        return self.output


###############  SOFTMAX  #############################


##################  Loss Computation  ###################

def loss(output, label):
    Loss = np.zeros(num_images)
    for p in range(num_images):
        photo = np.log(output[p])
        y = label[p][0]
        x = photo[y][0]
        Loss[p] = -x
    print(Loss)





##################  Loss Computation  ###################


################# input sample   ############################

with open('sample.txt','r') as file:
    line = file.readline()
    # print(line.split()[0])
    while line:
        # print(line)
        if line.split()[0]=="Conv":
            a = int(line.split()[1])
            b = int(line.split()[2])
            c = int(line.split()[3])
            d = int(line.split()[4])
            c1 = convolution(data,data.shape[2],b,a,d,c)
            data = c1.forward()
            print(data.shape)

        if line.split()[0]=="ReLU":
            r1 = relu(data)
            data = r1.forward()
            print(data.shape)

        if line.split()[0]=="Pool":
            e = int(line.split()[1])
            f = int(line.split()[2])
            p1 = pool(data,e,f)
            data = p1.forward()
            print(data.shape)

        if line.split()[0]=="FC":
            g = int(line.split()[1])
            f1 = full_connect(data,data.shape[1],g)
            data = f1.forward()
            print(data.shape)

        if line.split()[0]=="Softmax":
            s1 = softmax(data)
            data = s1.forward()
            print(data.shape)

        line = file.readline() 
    print("---------------------------------")
    

print(data)
loss(data,labels)  ##################  Loss function



