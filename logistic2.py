import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import statsmodels.api as sm
from patsy import dmatrices
from ggplot import *

bank=pd.read_csv('D:\\Yaoming\\Learn\\Python\\learning-python-predictive-analytics\\datasets/bank.csv',sep=';')
bank.head()

bank.columns.values

bank.dtypes

bank['y']=(bank['y']=='yes').astype(int)

bank['education'].unique()

bank['education'].value_counts()

#combine levels

bank['education']=np.where(bank['education']=='basic.9y','Basic', bank['education'])
bank['education']=np.where(bank['education']=='basic.6y','Basic', bank['education'])
bank['education']=np.where(bank['education']=='basic.4y','Basic', bank['education'])
bank['education']=np.where(bank['education']=='university.degree','University Degree', bank['education'])
bank['education']=np.where(bank['education']=='professional.course','Professional Course', bank['education'])
bank['education']=np.where(bank['education']=='high.school','High School', bank['education'])
bank['education']=np.where(bank['education']=='illiterate','Illiterate', bank['education'])
bank['education']=np.where(bank['education']=='unknown','Unknown', bank['education'])

bank['y'].value_counts()

bank.groupby('y').mean()
bank.groupby('education').mean()

pd.crosstab(bank.education,bank.y).plot(kind='bar')
plt.title('Purchase Frequency for Education Level')
plt.xlabel('Education')
plt.ylabel('Frequency of Purchase')

table=pd.crosstab(bank.marital, bank.y)
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')

# Create dummy variables for categorical variables

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list=pd.get_dummies(bank[var],prefix=var)
    bank1=bank.join(cat_list)
    bank=bank1
    
bank_vars=bank.columns.values.tolist()
to_keep=[i for i in bank_vars if i not in cat_vars]

bank_final=bank[to_keep]
bank_final.columns

bank_final_vars=bank_final.columns.values.tolist()
Y=['y']
X=[i for i in bank_final_vars if i not in Y]


#Using RFE to select the important features.
model=LogisticRegression()
rfe=RFE(model,12)
rfe=rfe.fit(bank[X],bank[Y])
print(rfe.support_)
print(rfe.ranking_)

rfe.support_.tolist()
rfe.ranking_.tolist()

selected1=pd.DataFrame(X,columns=['Name'])
selected1['selected']=rfe.ranking_.astype(int)
cols=selected1[selected1['selected']==1]['Name'].tolist()

#replace - by _ in cols
cols=[w.replace('-', '_') for w in cols]
cols

#change the bank_final column names with '-' to '_'
bank_colnames=bank_final.columns.values.tolist()
bank_colnames_new=[w.replace('-','_') for w in bank_colnames]
bank_final=bank_final.rename(columns=dict(zip(bank_colnames, bank_colnames_new)))

#X=bank_final[cols]
#Y=bank_final['y']

bank_final['y']=(bank_final['y']=='yes').astype(int)

#Implementing the model with statsmodel.api method
logit_model=sm.Logit(bank_final['y'],bank_final[cols])
result=logit_model.fit()
print (result.summary())

#scikit-learn method
X=bank_final[cols]
Y=bank_final['y']
clf=LogisticRegression()
clf.fit(X,Y)
clf.score(X,Y)
Y.mean()

print('intercept',clf.intercept_)
##get the values of the coefficients:
parameters=pd.DataFrame(list(zip(X.columns,np.transpose(clf.coef_))),columns=['Name','Coeff'])

parameters['Coeff']=parameters.Coeff.astype(float)

parameters

####### Model validation and evaluation
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
clf1=LogisticRegression()
result=clf1.fit(X_train,Y_train)
probs=clf1.predict_proba(X_test)
probs

predicted=clf1.predict(X_test)

#The changing of threshold values for classification
prob=probs[:,1]
prob_df=pd.DataFrame(prob)
prob_df['predict']=np.where(prob_df[0]>=0.10,1,0)
prob_df.head()
#accuracy of the model;
print('Model accuracy:',metrics.accuracy_score(Y_test,predicted))
print('AUC:', metrics.roc_auc_score(Y_test,probs[:,1]))

################### Cross Validation ##########################
scores=cross_val_score(LogisticRegression(),X,Y,scoring='accuracy',cv=10)
print (scores)
print (scores.mean())

###################### Model Validation: ROC curve
#del prob_df
prob_df=pd.DataFrame(prob)
prob_df['predict']=np.where(prob_df[0]>=0.05,1,0)
prob_df['actual']=Y_test.values
prob_df.head()

confusion_matrix=pd.crosstab(prob_df['actual'],prob_df['predict'])
confusion_matrix

Sensitivity=[1,0.95,0.87,0.62,0.67,0.59,0.5,0.41,0]
FPR=[1,0.76,0.62,0.23,0.27,0.17,0.12,0.07,0]
plt.plot(FPR,Sensitivity,marker='o',linestyle='--',color='r')
x=[i*0.01 for i in range(100)]
y=[i*0.01 for i in range(100)]
plt.plot(x,y)
plt.xlabel('(1-Specificity)')
plt.ylabel('Sensitivity')
plt.title('ROC Curve')

prob = clf1.predict_proba(X_test)[:,1]
fpr, sensitivity, _ = metrics.roc_curve(Y_test, prob)  #digout ROC data for plotting
df=pd.DataFrame(dict(fpr=fpr,sensitivity=sensitivity))
ggplot(df,aes(x='fpr',y='sensitivity')) + geom_line() + geom_abline(linetype='dashed') + xlim(0,1.1)+ylim(0,1.1)

#AUC:
auc=metrics.auc(fpr, sensitivity)
auc

ggplot(df, aes(x='fpr', y='sensitivity')) + \
geom_area(alpha=0.2) + geom_line(aes(y='sensitivity')) + \
ggtitle("ROC Curve w/ AUC=%s"  %str(auc))
