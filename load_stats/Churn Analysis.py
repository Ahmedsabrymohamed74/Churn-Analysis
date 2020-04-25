import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
size = 30000

# =============================================================================
#               Loading and cleaning data. w/ basic stats 
# =============================================================================
#Load the churn data file 
data= pd.read_csv("cell2celltrain.csv")

 #index data fields
Ax= data.iloc[:,:]
df= pd.DataFrame(Ax)

#Check&Clean data by removing nan
df = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

#Calculate and display the probability
Prizm = df.PrizmCode
prob = (Prizm.value_counts()/49752)

#Calculate the joint probability
joint_p = df.groupby(["HasCreditCard","PrizmCode"]).size()/49752

#Calculate the conditional probability
cond = joint_p.loc['Yes', 'Town']
cond_prob = cond/prob.Town

cp1 = df.groupby('Churn')['ChildrenInHH'].value_counts() / df.groupby('Churn')['ChildrenInHH'].count()

#Calculate the histogram
q = df.MonthlyRevenue
plt.hist(q, bins = 15,color='r')
plt.xlabel('MonthlyRevenue')
plt.ylabel('User')
plt.show()

# Calculate and display in a figure the PDF
hist,hist_edge = np.histogram(q,density=True,)
plt.xlabel('AgeHH1')
plt.ylabel('User')
plt.plot(hist)
plt.show()

#Calculate and display in a figure the CDF
cdf = np.cumsum(hist)
plt.plot(hist_edge[1:],cdf/cdf[-1])
plt.xlabel('AgeHH1')
plt.ylabel('User')
plt.show()

# Calculate the joint pdf
plt.hist2d(x=df.AgeHH1,y=df.AgeHH2,density=True,bins=(20,20),cmap='RdYlBu')
plt.xlabel('AgeHH1')
plt.ylabel('User')
plt.show()

#Calculate the mean and variance 
variance = df.var(axis=0)
mean = q.mean(axis=0)
print(mean)

# =============================================================================
# Finding relation and best fits considering anomalies. Calc bayes to find churn and test accuracy
# =============================================================================
     
         ####################  Covariance&Correlation ##################
         
#df = pd.DataFrame(np.random.randn(1000,2)), columns = ['MonthlyMinutes', 'MonthlyRevenue']
dy = df[["InboundCalls","OutboundCalls"]]
print('Covariance:')
covariance = dy.cov()
print(covariance)
print('___________' + '\n')
print('Correlation:')
correlation = dy.corr()
print(correlation)
print('___________' + '\n')

            ########################  Anomalies ########################
    #replacing missing values with mean
ddy = df["OutboundCalls"]
#plt.legend()
size = len(ddy.index)
columns = ["OutboundCalls"]
q1 = ddy.quantile(.25) #,interpolation = 'nearest'
q2 = ddy.quantile(.75)#,interpolation = 'nearest''
removed_outliers = ddy.between(q1,q2)
print(str(ddy[removed_outliers].size) + "/" + str(size) + "data points remain." + '\n') 
print('___________' + '\n')

plt.subplot(231)
ddy[removed_outliers].plot(label = "OutboundCalls",linewidth = 1/25).get_figure()
plt.legend(loc='upper right')

plt.subplot(232)
plt.plot(removed_outliers,label = "removed",linewidth = 1/25)
new_ddy = ddy - removed_outliers
plt.legend(loc='upper right')

plt.subplot(233)
plt.plot(new_ddy,label = "new OutboundCalls",linewidth = .5)
plt.legend(loc='upper right')


print(removed_outliers.value_counts())
print('___________' + '\n')


            ######################  fitting(PDF) ########################
			
df_cust = df["PeakCallsInOut"]
#df_cust.drop(["Churn"],inplace=True,axis=0)

size = len(df_cust.index)
AX = scipy.arange(size)
#Ay = scipy.int_(scipy.round_(scipy.stats.vonmises.rvs(5,size=size)*47))
#plt.subplot(231)
df_cust.plot(kind = 'hist',bins=[150,300,450,600],density=True,label = "Inbound")

plt.show()

dist_names = ['norm','expon','chi','invgamma','invgauss','chi2','norminvgauss']

for dist_name in dist_names:
    dist = getattr(scipy.stats, dist_name)
    param = dist.fit(df_cust)
    pdf_fitted = dist.pdf(AX, *param[:-2], loc=param[-2], scale=param[-1]) * size
    best_distribution = 'norm'
    best_params = (0.0, 1.0)
    best_sse = np.inf
    sse = np.sum(np.power(df_cust - pdf_fitted, 2.0))
    plt.plot(pdf_fitted, label=dist_name)
    plt.xlim(150,600)
    plt.ylim(0,100)
plt.legend(loc='upper right')
plt.figure(figsize=(12,8))
plt.show()


if best_sse > sse > 0:
        best_distribution = dist_name
        best_params = param
        best_sse = sse
      
#pdf = distribution.pdf(AX, loc=loc, scale=scale, *arg)


print("Error of best fit: ")
print(best_sse)


            ######################## Bayes ########################
# Create an empty data_bayesframe
data_bayes = df

# Create our target variable
data_bayes['Churn'] = df["Churn"]

# Create our feature variables
data_bayes['AgeHH1'] = df["AgeHH1"]
data_bayes['MonthlyRevenue'] = df["MonthlyRevenue"]
data_bayes['PercChangeRevenues'] = df["PercChangeRevenues"]

# View the data_bayes
print('Original data_bayes:' + '\n')
print(data_bayes)


# Create an empty data_bayesframe
Churn = df["Churn"]

# Create some feature values for this single row
Churn['AgeHH1'] = [50]
Churn['MonthlyRevenue'] = [71.99]
Churn['PercChangeRevenues'] = [15.9]

# View the data 
#Churn
# Number of males (bool = yes?) eh males dih???
n_yes = data_bayes['Churn'][data_bayes['Churn'] == 'Yes'].count()

# Number of males
n_no = data_bayes['Churn'][data_bayes['Churn'] == 'No'].count()

# Total rows
total_churn = data_bayes['Churn'].count()

# Number of Yes divided by the total rows
P_yes = n_yes/total_churn

# Number of No divided by the total rows
P_no = n_no/total_churn

# Group the data by Churn and calculate the means of each feature
Churn_means = data_bayes.groupby('Churn').mean()

# View the values
print('Churn Means:')
print(Churn_means)
print('___________' + '\n')


# Group the data by Churn and calculate the variance of each feature
data_variance = data_bayes.groupby('Churn').var()

# View the values
#data_variance

# Means for Yes
yes_AgeHH1_mean = Churn_means['AgeHH1'][data_variance.index == 'Yes'].values[0]
yes_MonthlyRevenue_mean = Churn_means['MonthlyRevenue'][data_variance.index == 'Yes'].values[0]
yes_PercChangeRevenues_mean = Churn_means['PercChangeRevenues'][data_variance.index == 'Yes'].values[0]

# Variance for Yes
yes_AgeHH1_variance = data_variance['AgeHH1'][data_variance.index == 'Yes'].values[0]
yes_MonthlyRevenue_variance = data_variance['MonthlyRevenue'][data_variance.index == 'Yes'].values[0]
yes_PercChangeRevenues_variance = data_variance['PercChangeRevenues'][data_variance.index == 'Yes'].values[0]

# Means for No
no_AgeHH1_mean = Churn_means['AgeHH1'][data_variance.index == 'No'].values[0]
no_MonthlyRevenue_mean = Churn_means['MonthlyRevenue'][data_variance.index == 'No'].values[0]
no_PercChangeRevenues_mean = Churn_means['PercChangeRevenues'][data_variance.index == 'No'].values[0]

# Variance for No bene7tag el mean wel variance f eh??
no_AgeHH1_variance = data_variance['AgeHH1'][data_variance.index == 'No'].values[0]
no_MonthlyRevenue_variance = data_variance['MonthlyRevenue'][data_variance.index == 'No'].values[0]
no_PercChangeRevenues_variance = data_variance['PercChangeRevenues'][data_variance.index == 'No'].values[0]

    # Create a function that calculates p(x | y):
def p_x_given_y(x, mean_y, variance_y):

    # Input the arguments into a probability density function
    p = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2)/(2*variance_y))
    
    # return p
    return p

# Numerator of the posterior if the unclassified observation is a Yes
P_yes = P_yes * \
p_x_given_y(Churn['AgeHH1'][0], yes_AgeHH1_variance, yes_AgeHH1_variance) * \
p_x_given_y(Churn['MonthlyRevenue'][0], yes_MonthlyRevenue_mean, yes_MonthlyRevenue_variance) * \
p_x_given_y(Churn['PercChangeRevenues'][0], yes_PercChangeRevenues_mean, yes_PercChangeRevenues_variance)
print('Calculated Bayes of Yes:')
print(P_yes)
print('\n')

# Numerator of the posterior if the unclassified observation is a No
P_no = P_no * \
p_x_given_y(Churn['AgeHH1'][0], no_AgeHH1_mean, no_AgeHH1_variance) * \
p_x_given_y(Churn['MonthlyRevenue'][0], no_MonthlyRevenue_mean, no_MonthlyRevenue_variance) * \
p_x_given_y(Churn['PercChangeRevenues'][0], no_PercChangeRevenues_mean, no_PercChangeRevenues_variance)
print('Calculated Bayes of No:')
print(P_no)
print('___________' + '\n')

                ###################### Accuracy ########################

# load the iris dataset 
from sklearn.datasets import load_iris 
iris = load_iris() 
  
# store the feature matrix (X) and response vector (y) 
X = iris.data 
y = iris.target 
  
# splitting X and y into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) 
  
# training the model on training set 
gnb = GaussianNB() 
gnb.fit(X_train, y_train) 
  
# making predictions on the testing set 
y_pred = gnb.predict(X_test) 
  
# comparing actual response values (y_test) with predicted response values (y_pred) 
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)