
import numpy as np
from numpy.core.fromnumeric import transpose
import scipy.stats as stats
import math
import copy

##################  data start  ########################
d=[]
with open('data.txt','r') as file:
    line = file.readline()
    while line:
        #print(line)
        d.append(float(line))
        line = file.readline() 

########################  data finish  #######################

######################### parameters start ########################
N = 0
trans = 0
#state = 0
mu = 0
sig = 0

with open('parameters.txt','r') as file:
   
    line = file.readline()
    n = int(line)
    N=n
    print(n)

    temp1 = np.zeros((n,n))
    temp2 = np.zeros(n)
    temp3 = np.zeros(n)
    #print(temp)

    for i in range(n):
        c = 0
        line = file.readline()
        for word in line.split():
            #print(word)
            temp1[i][c] = float(word)
            c = c+1
    trans = temp1
    #state = temp1
    
    line = file.readline()
    c=0
    for word in line.split():
        #print(word)
        temp2[c] = float(word)
        c = c+1
    mu = temp2

    line = file.readline()
    c=0
    for word in line.split():
        #print(word)
        temp3[c] = float(word)
        c = c+1
    sig = temp3


################ parameter finished ################

em = np.zeros((len(d),N))
prob = np.zeros((len(d),N))

def pre(trans,mu,sig):

    # state = copy.deepcopy(trans)
    trans = np.transpose(trans)
    for i in range(N):
        trans[i][i] = trans[i][i]-1

    x = np.full(N,1.0)
    temp = np.delete(trans,N-1,0)
    temp = np.vstack ((temp, x))
    a = np.array(temp)
    b = np.array([0,1])
    sv = np.linalg.solve(a, b)  #  lambda1 & lambda2 value
    # print(sv)

    ############## Emission matrix ##############

    # em = np.zeros((len(d),2))
    mu1 = mu[0]
    mu2 = mu[1]

    sig1 = sig[0]
    sig2 = sig[1]

    # for i in range(len(d)):
    #     em[i][0] = stats.norm.pdf(d[i], mu1, math.sqrt(sig1))
    #     em[i][1] = stats.norm.pdf(d[i], mu2, math.sqrt(sig2))
    for i in range(len(d)):
        for j in range(N):
            em[i][j] = stats.norm.pdf(d[i], mu[j], math.sqrt(sig[j]))
            # em[i][1] = stats.norm.pdf(d[i], mu2, math.sqrt(sig2))
        # em[i][0] = stats.norm.pdf(d[i], mu1, sig1)
        # em[i][1] = stats.norm.pdf(d[i], mu2, sig2)
    # a1 = math.log(sv[0]*em[0][0])
    # b1 = math.log(sv[1]*em[0][1])
    # prob[0][0] = a1
    # prob[0][1] = b1
    for i in range(N):
        prob[0][i] = math.log(sv[i]*em[0][i])

    return sv

    #############################################

  

store = np.zeros((len(d)-1,N)) # max cond store
road = [] # most probable path store

def backtrack(j,h):
    #if j==0:
    z = len(d)-2
    for i in range(z,-1,-1):
        if j==0:
            j = store[i][0]
            road.append(j)
        else:
            j = store[i][1]
            road.append(j)
    
    # print(road)

    for k in range(len(road)):
        if road[k] == 0:
            road[k] = '"El Nino"'
        else:
            road[k] = '"La Nina"'
    #print(road)
    road.reverse()
    
    if h == 0:
        with open('states_Viterbi_wo_learning.txt', 'w') as f:
            for line in road:
                f.write(line)
                f.write('\n')
    else:
        # print(len(road))
        with open('states_Viterbi_after_learning.txt', 'w') as f:
            for line in road:
                f.write(line)
                f.write('\n')


def viterbi(aa,ab,ba,bb,h):
    
    for i in range(1,len(d)):
        prob[i][0] = max( prob[i-1][0] + math.log(aa*em[i][0]) , prob[i-1][1] + math.log(ba*em[i][0]) )
        if prob[i-1][0] + math.log(aa*em[i][0]) > prob[i-1][1] + math.log(ba*em[i][0]):
            store[i-1][0] = 0
        else:
            store[i-1][0] = 1

        prob[i][1] = max( prob[i-1][0] + math.log(ab*em[i][1]) , prob[i-1][1] + math.log(bb*em[i][1]) )
        if prob[i-1][0] + math.log(ab*em[i][1]) > prob[i-1][1] + math.log(bb*em[i][1]):
            store[i-1][1] = 0
        else:
            store[i-1][1] = 1
    #print("x")
    if prob[len(d)-1][0] > prob[len(d)-1][1]:
        j = 0
    else:
        j = 1
    road.append(j)
    backtrack(j,h)

# viterbi()


############### E Step #################

fwd = np.zeros((2,len(d))) 
fsink = 0

# em = transpose(em)
# print(em.shape)
bwd = np.zeros((2,len(d)))
pi_star = np.zeros((2,len(d)))  
pi_2_star = np.zeros((N*N,len(d)-1)) 
Tn = np.zeros(N*N)
mean = np.zeros(N)
sd = np.zeros(N)

def forward(sv,aa,ab,ba,bb,em):

    em = transpose(em)
    f1 = em[0][0] * sv[0]
    f2 = em[1][0] * sv[1]

    fwd[0][0] = f1/(f1+f2)
    fwd[1][0] = f2/(f1+f2)

    for i in range(1,len(d)):
        fa = fwd[0][i-1]*aa*em[0][i] + fwd[1][i-1]*ba*em[0][i]
        fb = fwd[0][i-1]*ab*em[1][i] + fwd[1][i-1]*bb*em[1][i]
        fwd[0][i] = fa/(fa+fb)
        fwd[1][i] = fb/(fa+fb)
    fsink = fwd[0][len(d)-1] + fwd[1][len(d)-1] 

# fwd = transpose(fwd)
# np.savetxt('test.txt', fwd)

# forward(sv)
############### Forward Done ##################

############## Backward Matrix ##################

# bwd = np.zeros((2,len(d))) 

# def backward(aa,ab,ba,bb):
    b1 = 1
    b2 = 1
    bwd[0][len(d)-1] = b1/(b1+b2)
    bwd[1][len(d)-1] = b2/(b1+b2)

    z = len(d)-2
    for i in range(z,-1,-1):
        b3 = bwd[0][i+1]*aa*em[0][i+1] + bwd[1][i+1]*ab*em[1][i+1]
        b4 = bwd[0][i+1]*ba*em[0][i+1] + bwd[1][i+1]*bb*em[1][i+1]
        bwd[0][i] = b3/(b3+b4)
        bwd[1][i] = b4/(b3+b4)

# bwd = transpose(bwd)
# np.savetxt('test.txt', bwd)

# backward()

##################################################

############## Pi Star ###################
# pi_star = np.zeros((2,len(d))) 

# def pi_st():
    pi_star = np.zeros((2,len(d))) 
    pi_star = (fwd * bwd)/fsink
    # pi_star = np.multiply(fwd,bwd)

    for i in range(len(d)):
        pi1 = pi_star[0][i]
        pi2 = pi_star[1][i]
        pi_star[0][i] = pi1/(pi1+pi2)
        pi_star[1][i] = pi2/(pi1+pi2)
    # return pi_star

# pi_star = transpose(pi_star)
# np.savetxt('test.txt', pi_star)

# pi_star = pi_st()

###################################################

################# Pi Double Star ##################

# pi_2_star = np.zeros((N*N,len(d)-1)) 

# def pi_2_st():

    for i in range(len(d)-1):
        ps1 = (fwd[0][i] * aa * em[0][i+1] * bwd[0][i+1])/fsink
        ps2 = (fwd[0][i] * ab * em[1][i+1] * bwd[1][i+1])/fsink
        ps3 = (fwd[1][i] * ba * em[0][i+1] * bwd[0][i+1])/fsink
        ps4 = (fwd[1][i] * bb * em[1][i+1] * bwd[1][i+1])/fsink

        pi_2_star[0][i] = ps1/(ps1+ps2+ps3+ps4)
        pi_2_star[1][i] = ps2/(ps1+ps2+ps3+ps4)
        pi_2_star[2][i] = ps3/(ps1+ps2+ps3+ps4)
        pi_2_star[3][i] = ps4/(ps1+ps2+ps3+ps4)

# pi_2_star = transpose(pi_2_star)
# np.savetxt('test.txt', pi_2_star)

# pi_2_st()

#############################################################
########################  M Step  ######################################

# T1 = np.zeros((N*N,1)) 
# Tn = np.zeros(N*N)

# def Tr1():

    T1 = np.zeros(N*N)
    for i in range(N*N):
        r = 0
        for j in range(len(d)-1):
            r = r + pi_2_star[i][j]
        T1[i] = r

    T1 = T1.reshape(N,N)
    T2 = copy.deepcopy(T1)

    for i in range(N):
        for j in range(N):
            T1[i][j] = T2[i][j]/sum(T2[i])




########## Transition Done ###################

# mean = np.zeros(N)

# def mn():
    for i in range(N):
        s1 = 0.0
        s2 = 0.0
        for j in range(len(d)):
            s1 = s1 + (pi_star[i][j] * d[j])
            s2 = s2 + pi_star[i][j]
            
        # print(s2)
        mean[i] = s1/s2

#print(mean)

# mn()
############## Mean done #####################

############## Std deviation ################

# sd = np.zeros(N)

# def devtn(): # deviation
    for i in range(N):
        s1 = 0
        s2 = 0
        for j in range(len(d)):
            s1 = s1 + (pi_star[i][j] * pow(d[j] - mean[i],2))
            s2 = s2 + pi_star[i][j]
            # sd[i] = math.sqrt(s1/s2)
            sd[i] = s1/s2

    return sd,mean,T1

#####################################################

# sv = pre(trans,mu,sig)
# print(trans)
# print(mu)
# print(sig)
# print(sv)

def all(trans,mu,sig):
    aa = trans[0][0]
    ab = trans[0][1]
    ba = trans[1][0]
    bb = trans[1][1]
    sv = pre(trans,mu,sig)
    sd ,mean,T1=forward(sv,aa,ab,ba,bb,em)
    return sd,mean,T1,sv

# sd,mean,T1 = all(trans,mu,sig)
# print(T1)

def wo_viterbi(trans,mu,sig):
    aa = trans[0][0]
    ab = trans[0][1]
    ba = trans[1][0]
    bb = trans[1][1]
    sv = pre(trans,mu,sig)
    viterbi(aa,ab,ba,bb,0)

# wo_viterbi(trans,mu,sig)


while True:
    prevTr = copy.deepcopy(trans)
    prevMu = copy.deepcopy(mu)
    prevSig = copy.deepcopy(sig)
    sd,mean,T1,sv = all(trans,mu,sig)
    c = 0
    for i in range(N):
        c = c + abs(sd[i]-prevSig[i]) + abs(mean[i]-prevMu[i])
    
    if(c<0.00001):
        # print(T1)
        # print(mean)
        # print(sd)
        # print(sv)

        trans = T1
        sig = sd
        mu = mean
        with open('parameters_learned.txt', 'w') as f:
            f.write(str(N))
            f.write('\n')
            for i in range(N):
                for j in range(N):
                    f.write(str(T1[i][j]))
                    f.write('  ')
                f.write('\n')
            for k in range(N):
                f.write(str(mean[k]))  
                f.write('  ')  
            f.write('\n')
            for m in range(N):
                f.write(str(sd[m])) 
                f.write('  ')   
            f.write('\n')
            for p in range(N):
                f.write(str(sv[p])) 
                f.write('  ')   
            f.write('\n')


        aa = trans[0][0]
        ab = trans[0][1]
        ba = trans[1][0]
        bb = trans[1][1]
        sv = pre(trans,mu,sig)
        viterbi(aa,ab,ba,bb,1)
        break
    else:
        trans = T1
        sig = sd
        mu = mean
        continue


