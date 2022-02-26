import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd 
import csv

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



#************* 1st Dataset ************************
    
data_set= pd.read_csv('first.csv') 

#removing space from data
data_set = data_set.replace(r'^\s*$', np.nan, regex = True)
a = SimpleImputer(missing_values=np.nan, strategy='mean')
b = a.fit(data_set[['TotalCharges']])
data_set[['TotalCharges']] = b.transform(data_set[['TotalCharges']])

# standardization
scale = MinMaxScaler()
M = scale.fit_transform(data_set[['tenure','MonthlyCharges','TotalCharges']])
M = pd.DataFrame(M)
#print(type(M))

# Nominal categorical value
N = pd.get_dummies(data_set[['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']])
#print(N)

#true-false
P = pd.get_dummies(data_set[['gender','Partner','Dependents','PhoneService','PaperlessBilling','Churn']],drop_first=True)
#print(P)

#Concatenation
df = pd.concat([M,N,P],axis=1)
#print(df)

# Input-Output
x = df.iloc[:, range(39)]
y = df.iloc[:, [39]]
#print(x)

#******************************************************************

#*********************2nd Dataset***********************************

# data_set= pd.read_csv('second.csv') 


# # standardization
# scale = MinMaxScaler()
# M = scale.fit_transform(data_set[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']])
# M = pd.DataFrame(M)
# #print(type(M))

# # Nominal categorical value
# N = pd.get_dummies(data_set[['workclass','education','marital-status','occupation','relationship','race','native-country']])
# #print(N)

# #true-false
# P = pd.get_dummies(data_set[['sex','salary-scale']],drop_first=True)
# #print(P)

# #Concatenation
# df = pd.concat([M,N,P],axis=1)
# #print(df)

# # Input-Output
# x = df.iloc[:, range(107)]
# y = df.iloc[:, [107]]
# #print(y)

#********************************************************************

#*************** 3rd Dataset ***************************************

# data_set= pd.read_csv('third.csv') 

# # #removing space from data
# # data_set = data_set.replace(r'^\s*$', np.nan, regex = True)
# # a = SimpleImputer(missing_values=np.nan, strategy='mean')
# # b = a.fit(data_set[['TotalCharges']])
# # data_set[['TotalCharges']] = b.transform(data_set[['TotalCharges']])

# # standardization
# scale = MinMaxScaler()
# M = scale.fit_transform(data_set[["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]])
# M = pd.DataFrame(M)
# #print(type(M))

# # Nominal categorical value
# #N = pd.get_dummies(data_set[['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']])
# #print(N)

# #true-false
# P = pd.get_dummies(data_set[['Class']],drop_first=True)
# #print(P)

# #Concatenation
# df = pd.concat([M,P],axis=1)
# #print(df)

# # Input-Output
# x = df.iloc[:, range(30)]
# y = df.iloc[:, [30]]
# #print(x)

#*********************************************************************


# Test-Train

xtrain, xtest, ytrain, ytest = train_test_split( x, y, test_size = 0.2, random_state = 42)

xtrain = xtrain.to_numpy()
xtest = xtest.to_numpy()
ytrain = ytrain.to_numpy()
ytest = ytest.to_numpy()

xtrain = xtrain.transpose()
xtest = xtest.transpose()
ytrain = ytrain.transpose()
ytest = ytest.transpose()
#print(xtrain.shape)

#*****************************************

#***************Logistic regression*********************

def propagate(w, b, X, Y):
    m = X.shape[1]
    
    z = np.dot(w.T,X) + b
    A = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    A = (A+1)/2

    t1 = (Y-A) * (1-(A*A))
    dw = (-2 * np.dot(X, t1.T))/m
    db = (-2* np.sum(t1))/m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    
    grads = {"dw": dw,
             "db": db}
 
    return grads

#******************************************    

def optimize(w, b, X, Y, num, rate):
    
    for i in range(num):
        
        grads = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]
        
        w = w - (rate*dw)
        b = b - (rate*db)
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params

#**********************************************

def predict(w, b, X):
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    z = np.dot(w.T,X) + b
    A = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    A = (A+1)/2

    Y_prediction = (A >= 0.5) * 1.0
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

def initialize(dim):
    
    w = np.zeros((dim,1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b


def model(X_train, Y_train, X_test, Y_test, num_itr, l_rate):
   
    w,b = initialize(X_train.shape[0])

    para= optimize(w, b, X_train, Y_train, num_itr, l_rate)

    w = para["w"]
    b = para["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    tn1, fp1, fn1, tp1 = confusion_matrix(Y_prediction_test[0], Y_test[0]).ravel()
    accu1 = ((tp1 + tn1)/(tp1 + fp1 + fn1 + tn1))
    tpr1 = tp1/(tp1+fn1) 
    tnr1 = tn1/(tn1+fp1)
    ppv1 = tp1/(tp1+fp1)
    fdr1 = fp1/(fp1+tp1)
    f1_test = (2*ppv1*tpr1)/(ppv1+tpr1)

    tn2, fp2, fn2, tp2 = confusion_matrix(Y_prediction_train[0], Y_train[0]).ravel()
    accu2 = ((tp2 + tn2)/(tp2 + fp2 + fn2 + tn2))
    tpr2 = tp2/(tp2+fn2) 
    tnr2 = tn2/(tn2+fp2)
    ppv2 = tp2/(tp2+fp2)
    fdr2 = fp2/(fp2+tp2)
    f1_train = (2*ppv2*tpr2)/(ppv2+tpr2)

    print("***Test Result***")
    print("Accuracy-",accu1)
    print("True Positive Rate-",tpr1)
    print("True Negative Rate-",tnr1)
    print("Positive Predictive Value-",ppv1)
    print("False Discovery Rate-",fdr1)
    print("F1 Score-",f1_test)

    print("\n***Train Result***")
    print("Accuracy-",accu2)
    print("True Positive Rate-",tpr2)
    print("True Negative Rate-",tnr2)
    print("Positive Predictive Value-",ppv2)
    print("False Discovery Rate-",fdr2)
    print("F1 Score-",f1_train)
    
    


model(xtrain, ytrain, xtest, ytest, 1000, 0.05)


#**************************************************************************************************









