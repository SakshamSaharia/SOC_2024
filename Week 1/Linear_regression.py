import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
pd.set_option('future.no_silent_downcasting', True)
file_name = 'lifeexpectancydataset.csv'
file_path = Path.cwd() / file_name
# file_path='C:\Users\saksh\Downloads\soc\linear reg\Copy of LifeExpectancyDataset - Sheet1.csv'


#degree of polynomial used for linear regression 
degree=2
#learning rate used for model 
learning_rate=0.01
#defining tolerance for the error in model
tolerance=10**-4

print("Estimated time taken: 12 seconds")
def read_csv_to_2d_array(file_path):
    df4 = pd.read_csv(file_path)
    df2=df4.copy()
    df=df2.dropna().copy()
    df.drop(['Country'], axis=1, inplace =True)
    
    # df['Country'] = pd.factorize(df['Country'])[0] + 1
    # df.loc[:, 'Country'] = pd.factorize(df['Country'])[0] + 1
    # print(rdata[90])
    maxv=df['Life Expectancy'].max()
    # print(maxv)
    column_to_move = 'Life Expectancy'
    
# Rearranging the columns
# Get a list of columns excluding the one to move
    df.columns = df.columns.str.strip()
    if column_to_move not in df.columns:
        raise ValueError(f"Column '{column_to_move}' not found in the DataFrame")
    df2=df
    # assigning 0 to developing and 1 to developed 
    df2['Status'] = df['Status'].replace('Developing', 0).infer_objects(copy=False)
    df3=df2
    df3['Status'] = df2['Status'].replace('Developed', 1).infer_objects(copy=False)
    
    df=df3
    # column_to_move = 'Country'
    # columns = [col for col in df.columns if col != column_to_move]
    column_to_move = 'Life Expectancy'
    columns = [col for col in df.columns if col != column_to_move]
# Append the column to move to the end of the list
    columns.append(column_to_move)
    

# Subtract the mean of each column from the corresponding entries
    
    df = df[columns]
    df=df.astype(float)
    rdata=df.values.tolist()
    column_means = df.mean()
    df= df - column_means
    ndata=df.values.tolist()
    # print(ndata[90])
    column_means=column_means.tolist()
    max_values = df.max()
    max_values=max_values.tolist()
    # Normalize each column by its maximum value
    df = df / max_values
    # max=df.apply(lambda x: x.max(), axis=0)
    # df = df.apply(lambda x: x / x.max(), axis=0)
# Reorder the DataFrame
    
    data = df.values.tolist()
    # print(data[30])
    max_values=np.array(max_values)
    column_means=np.array(column_means)
    
    return data,max_values,rdata,ndata,column_means

# Example usage
def estimator(w,b,x,y,a,max_values,rdata,tolerance):
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
            # np.array(data[i][:n], dtype=float)
            jf=np.dot(w,np.array(x[i], dtype=float))+b-y[i]
            # print(jf)
            dj_db+=jf/m
            dj_dw+=(jf/m)*np.array(x[i])
            
        # print(3)
        
        w=w-a*dj_dw
        b=b-a*dj_db
        
        if np.linalg.norm(dj_dw) < tolerance and abs(dj_db) < tolerance:
            # print("broke")
            break
    rsme=0
    for i in range(m):
        rjf=np.dot(w,np.array(x[i], dtype=float))+b-y[i]
        # if i<32 and i>29:
        #     print(rjf*max_values[-1])
        rsme+=rjf*rjf
    rsme=(rsme/m)**0.5
    rsme=rsme*max_values[-1]
   
    return w,b,rsme


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


data,max_values,rdata,ndata,column_means= read_csv_to_2d_array(file_path)
# x = np.array([[1], [2], [3], [4], [5.1], [6], [7], [8], [9], [10]])
# y = np.array([[2.1], [2.9], [3.8], [4.4], [5.1], [5.9], [6.8], [7.4], [8.2], [9.0]])

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

rsme=0
w,b,rsme=estimator(w,0,x,y,learning_rate,max_values,rdata,tolerance)
print("The rsme loss for model is: ",rsme)



def predictor(w,b,input,column_means,x,max_values):
    return (np.dot(w,np.array(input))+b)*max_values[-1]+column_means[-1]


# predicting value of life expectancy 
# for sample the data at 101th position is checked 
print("For sample the data at 101th position after dropping NA in dataset is checked , input variable can be changed as per choice")
input=rdata[100][:-1]
print("Expected Value is :", rdata[100][-1])

input=np.array([np.divide(input-column_means[:-1],max_values[:-1])])
input=generate_polynomial_features(input,degree)
input=input.reshape(-1)

print("Predicted Value is : ",predictor(w,b,input,column_means,x,max_values))
