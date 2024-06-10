import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
pd.set_option('future.no_silent_downcasting', True)
file_name = 'framingham.csv'
file_path = Path.cwd() / file_name
# file_path='C:\Users\saksh\Downloads\soc\linear reg\Copy of LifeExpectancyDataset - Sheet1.csv'
learning_rate2=0.01
cutoff2=30
learning_rate1=1
cutoff1=1
tolerance=10**-3
degree=2
print("Estimated Time for Model1 : 30 seconds")
print("Estimated Time for both Models : 60 seconds")

def read_csv_to_2d_array(file_path):
    df2 = pd.read_csv(file_path)
    df=df2.copy()

    # filling the NA values instead of deleting in dataset to increase accuracy
    df.loc[(df['currentSmoker'] == 0) & (df['cigsPerDay'].isnull()), 'cigsPerDay'] =0
    df['BPMeds']=df['BPMeds'].fillna(df['BPMeds'].mode()[0])
    df['totChol']=df['totChol'].fillna(df['totChol'].median())
    df['BMI']=df['BMI'].fillna(df['BMI'].median())
    df['heartRate']=df['heartRate'].fillna(df['heartRate'].median())
    df['glucose']=df['glucose'].fillna(df['glucose'].median())
    df=df.dropna()
    df=df.copy()

    df.drop(['education'], axis=1, inplace =True)
    
    
    # df.drop(['male'], axis=1, inplace =True)
# Define the element to drop and the column to check
    element_to_drop = 'NA'
    column_name = 'TenYearCHD'

# Filter the DataFrame to exclude rows with the element in the specified column
    df_filtered = df[df[column_name] != element_to_drop]
    df=df_filtered
   
    rdata=df.values.tolist()
    
    maxv=df['TenYearCHD'].max()
    

    df=df.astype(float)
    column_means = df.mean()
    df= df - column_means
    max_values = df.max()
    
    # Normalize each column by its maximum value
    df = df / max_values
    # max=df.apply(lambda x: x.max(), axis=0)
    # df = df.apply(lambda x: x / x.max(), axis=0)
# Reorder the DataFrame
    
    data = df.values.tolist()
    # print(data[30])
    column_means=column_means.tolist()
    column_means=np.array(column_means)
    max_values=max_values.tolist()
    return data,max_values,rdata,column_means

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Example usage
def estimator(w,b,x,y,a,max_values,rdata,cutoff,tolerance):
    m=x.shape[0]
    n=x.shape[1]
    j=0
    # print(2)
    while j<1000:
        j+=1 
        # print(w,b)
        # print(j)
        dj_dw2 = np.zeros(n, dtype=float)
        dj_dw=dj_dw2.astype(float)
        dj_db=0
        # print(1)
        for i in range(m):
            # jf=np.dot(w,np.array(data[i][:n], dtype=float))+b-data[i][n]
            jf=sigmoid(np.dot(w,np.array(x[i], dtype=float))+b)-y[i]
            # print(jf)
            dj_db+=jf/m
            dj_dw+=(jf/m)*x[i]
            
        # print(3)
        
        w=w-a*dj_dw
        b=b-a*dj_db
        

        # tolerance=10**-3
        if np.linalg.norm(dj_dw) < tolerance and abs(dj_db) < tolerance:
            print("broke")
            break
    fp=0
    fn=0
    tp=0
    tn=0
    fwbh=np.zeros(m)
    for i in range(m):    
        # jf=np.dot(w,np.array(data[i][:n], dtype=float))+b-data[i][n]
        f_wb=1000*sigmoid(np.dot(w,np.array(x[i], dtype=float))+b)
        fwbh[i]=f_wb
        # print(f_wb)
        if f_wb > cutoff and rdata[i][-1]==0 :
            fp+=1
        if f_wb <cutoff and rdata[i][-1]==1 :
            fn+=1
        if rdata[i][-1]==0:
            tn+=1
        if rdata[i][-1]==1:
            tp+=1

   
    return w,b,fp,fn,tp,tn,fwbh



# Generate polynomial features manually 
def generate_polynomial_features(X, degree):
    n_samples, n_features = X.shape
    X_poly = np.ones((n_samples, 1))  # Start with a column of ones for the bias term
    
    for d in range(1, degree + 1):
        for i in range(n_features):
            for j in range(i, n_features):
                if d == 1:
                    X_poly = np.hstack((X_poly, X[:, [i]]))
                else:
                    X_poly = np.hstack((X_poly, (X[:, [i]] * X[:, [j]])**(d-1)))

    return X_poly


data,max_values,rdata,column_means= read_csv_to_2d_array(file_path)

m=len(data)
n=len(data[0])-1
data=np.array(data)
# print(data[0])

x=data[:,:n]
# print(x[20])
y=data[:,n]
x=generate_polynomial_features(x,degree)
n=x.shape[1]
w = np.zeros(n)
# print(n,x.shape[1],w.shape)


w,b,fp,fn,tp,tn,fwbh=estimator(w,0,x,y,learning_rate1,max_values,rdata,cutoff1,tolerance)

print("Accuracy of Model1 : ",100*(1-(fp+fn)/(tp+tn)))
print("Percentage of False Positives in Model1 : ",100*(fp/tn))
print("Percentage of False Negatives in Model1 : ",100*(fn/tp))
# print(w,b,fp,fn,m,tp,tn,100*(fp/tn),100*(fn/tp))

w = np.zeros(n)
w,b,fp,fn,tp,tn,fwbh=estimator(w,0,x,y,learning_rate2,max_values,rdata,cutoff2,tolerance)
# for learning_rate in range(20):
#     w,b,fp,fn,tp,tn=estimator(w,0,data,learning_rate,max_values,rdata)
#     print(learning_rate,fp,fn)
print("Accuracy of Model2 : ",100*(1-(fp+fn)/(tp+tn)))
print("Percentage of False Positives  in Model2 : ",100*(fp/tn))
print("Percentage of False Negatives  in Model2 : ",100*(fn/tp))

# print(fwbh.size)
# y=data[:, -1
# # print(y.size)
# plt.scatter(fwbh,data[:,-1],color='red',label='data points')
# # plt.plot(fwbh,w*x+b,color='blue',label='reg line')
# plt.show()



def predictor(w,b,input,cutoff,column_means,x):
    if 1000*sigmoid(np.dot(w,np.array(input))+b)>cutoff:
        return 1
    else :
        return 0 

# for sample the data at 101th position is checked 
print("For sample the data at 101th position after dropping NA in dataset is checked , input variable can be changed as per choice")    

input=rdata[-100][:-1]
print("Expected Value : ",rdata[100][-1])


input=np.array([np.divide(input-column_means[:-1],max_values[:-1])])
input=generate_polynomial_features(input,degree)
input=input.reshape(-1)
print("Predicted Value : ",predictor(w,b,input,cutoff2,column_means,x))