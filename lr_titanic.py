# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 18:46:59 2019

@author: HP
"""
# libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report as cr
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score

# import data
path="C:/Users/HP/Desktop/PYTHON/Logistic_Regression/titanic.csv"
titanic = pd.read_csv(path)
pd.set_option("display.expand_frame_repr",False)

# check data details
titanic.head(5)
col = list(titanic.columns)
print(col)
titanic.shape
len(titanic.index)
titanic.head()
titanic.dtypes
titanic.describe
# rename the variables
titanic.columns

# drop pid variable
titanic = titanic.drop('Cabin',axis=1)
titanic = titanic.drop('Name',axis=1)
titanic = titanic.drop('Ticket',axis=1)



col = titanic.columns

# factor variables

factor_x = titanic.select_dtypes(exclude=["int64","float64","category"]).columns.values
print(factor_x)

#unique factor values
for c in factor_x:
    print("Factor variable = '" + c + "'")
    print(titanic[c].unique())
    print("***")

# check null values
for c in col:
    if (len(titanic[c][titanic[c].isnull()])) > 0:
        print("WARNING: Column '{}' has NULL values".format(c))

    if (len(titanic[c][titanic[c] == 0])) > 0:
        print("WARNING: Column '{}' has value = 0".format(c))
        
    if (len(titanic[c][titanic[c].isna()])) > 0:
        print("Na values in column '{}'".format(c))
        
# Na handling
titanic = titanic.fillna(titanic.mean())
titanic.head()
        
# 
for c in factor_x:
    titanic[c] = titanic[c].astype('category',copy=False)
titanic.dtypes

#rearrange columns
titanic= titanic[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']]

titanic.columns
    
#EDA
titanic['Survived'].value_counts()
titanic['Survived'].value_counts()/len(titanic)

# plot the "0" and "1"
# --------------------------------------
sns.countplot(x='Survived', data=titanic, palette='hls')

# group avg
titanic.groupby('Survived').mean()

# visualization
# Survived vs Sex
pd.crosstab(titanic.Sex, titanic.Survived).plot(kind='bar')
plt.title('Survived by Sex')
plt.xlabel('Gender')
plt.ylabel('Frequency of Survived')

titanic.columns
# Survived vs Pclass
pd.crosstab(titanic.Pclass, titanic.Survived).plot(kind='bar')
plt.title('Survived by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Frequency of Survived')

# Survived vs parch
pd.crosstab(titanic.Parch, titanic.Survived).plot(kind='bar')
plt.title('Survived by Pclass')
plt.xlabel('Parch')
plt.ylabel('Frequency of Survived')



# Survived vs Embarked
pd.crosstab(titanic.Embarked, titanic.Survived).plot(kind='bar')
plt.title('Survived by Pclass')
plt.xlabel('Embarked')
plt.ylabel('Frequency of Survived')

# proportion
# --------------------------------------
table=pd.crosstab(titanic.Sex, titanic.Survived)
table.div(table.sum(1).astype(float), 
          axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Sex vs Survived')
plt.xlabel('Sex')
plt.ylabel('Proportion of Survived')

# Pclass vs Survived
pd.crosstab(titanic.Pclass,titanic.Survived).plot(kind='bar')
plt.title('Pclass vs Survived')
plt.xlabel('Pclass')
plt.ylabel('Proportion of urvived')

# get dummy variable
titanic.Sex.unique()
pd.get_dummies(titanic.Sex,drop_first=True).tail(20)


#
print(factor_x)

new_titanic = titanic.copy()
#
for var in factor_x:
    cat_list = pd.get_dummies(titanic[var], drop_first=True, prefix=var)
    # data1=bank.join(cat_list)
    new_titanic = new_titanic.join(cat_list)
    
new_titanic


    
#
new_col_set = new_titanic.columns
print(new_col_set)
len(new_col_set)
new_titanic.head()   

new_titanic= new_titanic[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Sex_male', 'Embarked_Q', 'Embarked_S','Survived']]
new_titanic.columns#

#
to_keep = list(set(new_col_set).difference(set(factor_x)))
to_keep
to_keep.sort()
to_keep
len(to_keep)
    
#
titanic_final = new_titanic[to_keep]
titanic_final.head(4)
titanic_final.columns.sort_values()
len(titanic_final.columns)

titanic_final.columns
titanic_final = pd.concat(
        [titanic_final['Survived'], 
        titanic_final.drop('Survived',axis=1)],
        axis=1)
#
train, test = train_test_split(titanic_final, test_size = 0.3)

print(train.shape)
print(test.shape)

total_cols=len(titanic_final.columns)
print(total_cols)

train_x = train.iloc[:,1:total_cols+1]
train_y = train.iloc[:,0]

test_x  = test.iloc[:,1:total_cols+1]
test_y = test.iloc[:,0]


train_x.iloc[0:10]

# model
new_titanic.Survived.unique()
logit_model = sm.Logit(train_y, train_x)
logit_result = logit_model.fit()
logit_result.summary2()

#
kf = KFold(n_splits=5) 
kf.get_n_splits(train_x)
print(kf)

fold = 1
# split the training further into train and test
for train_index, test_index in kf.split(train_x):
    
    cv_train_x = train_x.iloc[train_index,]
    cv_train_y = train_y.iloc[train_index,]
    
    cv_test_x = train_x.iloc[test_index,]
    cv_test_y = train_y.iloc[test_index,]
    
    # build the model on the CV training data and predict on CV testing data
    cv_logit_model = sm.Logit(cv_train_y, cv_train_x).fit()
    cv_pdct = list(cv_logit_model.predict(cv_test_x))

    # set the default cut-off to 0.5
    # and set predictions to 0 and 1
    cv_length = len(cv_pdct)
    cv_results=list(cv_pdct).copy()
        
    for i in range(0,cv_length):
        if cv_pdct[i] <= 0.5:
            cv_results[i] = 0
        else:
            cv_results[i] = 1
    
    # accuracy score
    acc_score = accuracy_score(cv_test_y,cv_results)*100
    print('Fold={0},Accuracy={1}'.format(fold,acc_score) )
    
    fold+=1

##
pred_y = logit_result.predict(test_x)
y_results = list(pred_y)
length = len(y_results)

# set the default cut-off to 0.5
# and set predictions to 0 and 1
for i in range(0,length):
    if y_results[i] <= 0.5:
        y_results[i] = 0
    else:
        y_results[i] = 1
        
# accuracy score
print(accuracy_score(test_y,y_results)*100)

# confusion matrix
cm=ConfusionMatrix(list(y_results),list(test_y))
print(cm)
cm.print_stats()

# Classification report : precision, recall, F-score
print(cr(test_y, y_results))

# draw the ROC curve
from sklearn import metrics
import matplotlib.pyplot as plt

fpr, tpr, threshold = metrics.roc_curve(test_y, y_results)
roc_auc = metrics.auc(fpr,tpr)
print(roc_auc)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#  feature selection

# decision tree model
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.externals.six import StringIO 
from IPython.display import Image
# !pip install pydotplus
import pydotplus
# !pip install pandas_ml
from pandas_ml import ConfusionMatrix


# decision tree algorithm
clf_gini = dtc(criterion = "gini", random_state = 100, 
               max_depth=3, min_samples_leaf=5)

fit1 = clf_gini.fit(train_x, train_y)
print(fit1)

#plot
dot_data = StringIO()

tree.export_graphviz(fit1, out_file=dot_data,  
                filled=True, rounded=True, special_characters=True)

graph=pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())

#
pred_gini = fit1.predict(test_x)
pred_gini
len(test_y)
len(pred_gini)
print("Gini Accuracy is ", 
      accuracy_score(test_y,pred_gini)*100)

# create dataframe with the actual and predicted results
# -------------------------------------------------------
df_results1 = pd.DataFrame({'actual':test_y, 
                            'predicted':pred_gini})
df_results1

# another nice way to plot the results
# -------------------------------------
cm1=ConfusionMatrix(list(test_y), list(pred_gini))
# plot
# -------------------------------------
cm1
cm1.plot()
cm1.print_stats()

# model 2
# DT with Entropy(Information Gain) criteria
# ----------------------------------------------------
clf_entropy=dtc(criterion="entropy", 
                random_state=100, max_depth=3, 
                min_samples_leaf=5)

fit2 = clf_entropy.fit(train_x,train_y)
print(fit2)

pred_entropy = fit2.predict(test_x)

pred_entropy
len(test_x)
len(pred_entropy)
print("Entropy Accuracy is ", 
      accuracy_score(test_y,pred_entropy)*100)

df_results2 = pd.DataFrame({'actual':test_y, 
                            'predicted':pred_entropy})
df_results2

# another nice way to plot the results
# -------------------------------------
cm2=ConfusionMatrix(list(test_y), list(pred_entropy))
# plot
# -------------------------------------
cm2
cm2.plot()
cm2.print_stats()


# REF
from sklearn.feature_selection import RFE
rfe = RFE(fit1, 5)
rfe = rfe.fit(test_x, test_y)
support = rfe.support_
ranking = rfe.ranking_
df_rfe = pd.DataFrame({"columns":col[0:9], 
                       "support":support, 
                       "ranking":ranking})
df_rfe.sort_values("ranking")

# feature selction
from sklearn.feature_selection import f_regression as fs
X=train_x.iloc[:,0:11]
features = fs(X,train_y,center=True)
features[0]

list(features[0])

df_features = pd.DataFrame({"columns":train_x.columns[0:11], 
                            "score":features[0],
                            "p-val":features[1]
                            })
print(df_features)

# sort on columns
df_features.sort_values(['score'],ascending=False)















