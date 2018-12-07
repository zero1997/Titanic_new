# -*- coding: UTF-8 -*-
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import warnings
from sklearn import cross_validation
#忽略警告错误

#不用写循环，可以之间写函数进行列处理
def Fam_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0

def predict_age_use_cross_validationg(df1, df2, dfTest):
    age_df1 = df1[['Age', 'Pclass', 'Sex', 'Title']]
    age_df1 = pd.get_dummies(age_df1)
    age_df2 = df2[['Age', 'Pclass', 'Sex', 'Title']]
    age_df2 = pd.get_dummies(age_df2)
    known_age = age_df1[age_df1.Age.notnull()].as_matrix()
    unknown_age_df1 = age_df1[age_df1.Age.isnull()].as_matrix()
    unknown_age = age_df2[age_df2.Age.isnull()].as_matrix()
    y = known_age[:, 0]
    X = known_age[:, 1:]
    rfr = RandomForestRegressor(random_state = 0, n_estimators = 100, n_jobs = -1)
    rfr.fit(X, y)
    predictedAges = rfr.predict(unknown_age[:, 1::])
    df2.loc[(df2.Age.isnull()), 'Age'] = predictedAges
    predictedAges = rfr.predict(unknown_age_df1[:, 1::])
    df1.loc[(df1.Age.isnull()), 'Age'] = predictedAges
    age_Test = dfTest[['Age', 'Pclass', 'Sex', 'Title']]
    age_Test = pd.get_dummies(age_Test)
    age_Tmp = df2[['Age', 'Pclass', 'Sex', 'Title']]
    age_Tmp = pd.get_dummies(age_Tmp)
    age_Tmp = pd.concat([age_Test[age_Test.Age.notnull()], age_Tmp])
    known_age1 = age_Tmp.as_matrix()
    unknown_age1 = age_Tmp.as_matrix()
    unknown_age1 = age_Test[age_Test.Age.isnull()].as_matrix()
    y = known_age1[:, 0]
    x = known_age1[:, 1:]
    rfr.fit(x, y)
    predictedAges = rfr.predict(unknown_age1[:, 1:])
    dfTest.loc[(dfTest.Age.isnull()), 'Age'] = predictedAges
    return dfTest

warnings.filterwarnings("ignore") 

#取出数据并给年龄界定类型
train = pd.read_csv("train.csv", dtype={'Age': np.float64})
test = pd.read_csv("test.csv", dtype = {"Age": np.float64})
PassengerId = test['PassengerId']
all_data = pd.concat([train, test], ignore_index=True)
#train.info()

#性别与是否获救的关系图
#sns.barplot(x = "Sex", y = "Survived", data=train)
#plt.show()

#船舱等级与是否幸存的关系
#sns.barplot(x="Pclass", y = "Survived", data=train)
#plt.show()
'''
#年龄的影响
facet = sns.FacetGrid(train, hue="Survived", aspect=2)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.show()
'''
#兄弟姐妹的影响
#sns.barplot(x="SibSp", y="Survived", data=train)
#plt.show()

#父母子女的影响
#sns.barplot(x="Parch", y="Survived", data=train)
#plt.show()

#train.loc[train.Cabin.isnull(), 'Cabin'] = 'U0'

#create Deck
all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
all_data['Deck'] = all_data['Cabin'].str.get(0)

#Title
all_data["Title"] = all_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
all_data['Title'] = all_data['Title'].map(Title_Dict)
#print all_data
#sns.barplot(x="Title", y="Survived", data=all_data)
#plt.show()

#create FamilySize&FamilyLabel
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data['FamilyLabel'] = all_data['FamilySize'].apply(Fam_label)

#create TicketGroup 
Ticket_Count = dict(all_data['Ticket'].value_counts())
all_data['TicketGroup'] = all_data['Ticket'].apply(lambda x: Ticket_Count[x])


#通过构造随机森林回归决策树模型填充Age
train = all_data[all_data['Survived'].notnull()]
test = all_data[all_data['Survived'].isnull()]
train_split_1, train_split_2 = cross_validation.train_test_split(train, test_size=0.5, random_state=0)
t1 = train_split_1.copy()
t2 = train_split_2.copy()
tmp1 = test.copy()
t5 = predict_age_use_cross_validationg(t1,t2,tmp1)
t1 = pd.concat([t1,t2])
t3 = train_split_1.copy()
t4 = train_split_2.copy()
tmp2 = test.copy()
t6 = predict_age_use_cross_validationg(t4,t3,tmp2)
t3 = pd.concat([t3,t4])
train['Age'] = (t1['Age'] + t3['Age'])/2
test['Age'] = (t5['Age'] + t6['Age']) / 2
all_data = pd.concat([train,test])
#print all_data.describe()

#填充Embarked
all_data['Embarked'] = all_data['Embarked'].fillna('C')
#填充Fare
fare = all_data[(all_data['Embarked'] == 'S') & (all_data['Pclass'] == 3)].Fare.median()
all_data['Fare'] = all_data['Fare'].fillna(fare)

#同组识别
all_data['Surname']=all_data['Name'].apply(lambda x:x.split(',')[0].strip())
Surname_Count = dict(all_data['Surname'].value_counts())
all_data['FamilyGroup'] = all_data['Surname'].apply(lambda x:Surname_Count[x])
Female_Child_Group=all_data.loc[(all_data['FamilyGroup']>=2) & ((all_data['Age']<=12) | (all_data['Sex']=='female'))]
Male_Adult_Group=all_data.loc[(all_data['FamilyGroup']>=2) & (all_data['Age']>12) & (all_data['Sex']=='male')]
Female_Child=pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
Female_Child.columns=['GroupCount']
Male_Adult=pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())
Male_Adult.columns=['GroupCount']
Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()
Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()
Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)
train=all_data.loc[all_data['Survived'].notnull()]
test=all_data.loc[all_data['Survived'].isnull()]
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 'male'
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Age'] = 60
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Title'] = 'Mr'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 'female'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Title'] = 'Miss'


train.to_csv("train1.csv")
test.to_csv("test1.csv")





