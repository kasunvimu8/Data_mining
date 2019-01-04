import pandas
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

filename = 'breaset-cancer.csv'
data = pandas.read_csv(filename, encoding = "ISO-8859-1")
#print(data.shape) #make sure the whole data is being read
data = data.dropna()  # records with missing values are dropped here
#print(data.shape) # data without missing values
#print(data)

data.drop(['COUNTRY'], axis=1, inplace=True)

# BREASTCANCERPER100TH conversion
data.BREASTCANCERPER100TH.iloc[0:] [data.BREASTCANCERPER100TH.iloc[0:]<20 ] = 0
data.BREASTCANCERPER100TH.iloc[0:] [data.BREASTCANCERPER100TH.iloc[0:]>=20 ] = 1

# INCOMEPERPERSON conversion
data.INCOMEPERPERSON.iloc[0:] [data.INCOMEPERPERSON.iloc[0:]<2000 ] = 0
data.INCOMEPERPERSON.iloc[0:] [data.INCOMEPERPERSON.iloc[0:]>=2000 ] = 1

# ALCCONSUMPTION conversion
data.ALCCONSUMPTION.iloc[0:] [data.ALCCONSUMPTION.iloc[0:]<5 ] = 0
data.ALCCONSUMPTION.iloc[0:] [data.ALCCONSUMPTION.iloc[0:]>=5 ] = 1

# ARMEDFORCESRATE conversion
data.ARMEDFORCESRATE.iloc[0:] [data.ARMEDFORCESRATE.iloc[0:]<0.8 ] = 0
data.ARMEDFORCESRATE.iloc[0:] [data.ARMEDFORCESRATE.iloc[0:]>=0.8 ] = 1

# CO2EMISSIONS conversion
data.CO2EMISSIONS.iloc[0:] [data.CO2EMISSIONS.iloc[0:]<1.000000e+09 ] = 0
data.CO2EMISSIONS.iloc[0:] [data.CO2EMISSIONS.iloc[0:]>=1.000000e+09 ] = 1

# FEMALEEMPLOYRATE conversion
data.FEMALEEMPLOYRATE.iloc[0:] [data.FEMALEEMPLOYRATE.iloc[0:]<30 ] = 0
data.FEMALEEMPLOYRATE.iloc[0:] [data.FEMALEEMPLOYRATE.iloc[0:]>=30 ] = 1

# HIVRATE conversion
data.HIVRATE.iloc[0:] [data.HIVRATE.iloc[0:]<0.5 ] = 0
data.HIVRATE.iloc[0:] [data.HIVRATE.iloc[0:]>=0.5 ] = 1

# INTERNETUSERATE conversion
data.INTERNETUSERATE.iloc[0:] [data.INTERNETUSERATE.iloc[0:]<40 ] = 0
data.INTERNETUSERATE.iloc[0:] [data.INTERNETUSERATE.iloc[0:]>=40 ] = 1

# LIFEEXPECTANCY conversion
data.LIFEEXPECTANCY.iloc[0:] [data.LIFEEXPECTANCY.iloc[0:]<70 ] = 0
data.LIFEEXPECTANCY.iloc[0:] [data.LIFEEXPECTANCY.iloc[0:]>=70 ] = 1

# OILPERPERSON conversion
data.OILPERPERSON.iloc[0:] [data.OILPERPERSON.iloc[0:]<1 ] = 0
data.OILPERPERSON.iloc[0:] [data.OILPERPERSON.iloc[0:]>=1 ] = 1

# POLITYSCORE conversion
data.POLITYSCORE.iloc[0:] [data.POLITYSCORE.iloc[0:]<=0 ] = 0
data.POLITYSCORE.iloc[0:] [data.POLITYSCORE.iloc[0:]>0 ] = 1

# RELECTRICPERPERSON conversion
data.RELECTRICPERPERSON.iloc[0:] [data.RELECTRICPERPERSON.iloc[0:]<500 ] = 0
data.RELECTRICPERPERSON.iloc[0:] [data.RELECTRICPERPERSON.iloc[0:]>=500 ] = 1

# SUICIDEPER100TH conversion
data.SUICIDEPER100TH.iloc[0:] [data.SUICIDEPER100TH.iloc[0:]<7 ] = 0
data.SUICIDEPER100TH.iloc[0:] [data.SUICIDEPER100TH.iloc[0:]>=7 ] = 1

# EMPLOYRATE conversion
data.EMPLOYRATE.iloc[0:] [data.EMPLOYRATE.iloc[0:]<50 ] = 0
data.EMPLOYRATE.iloc[0:] [data.EMPLOYRATE.iloc[0:]>=50 ] = 1

# urbanrate conversion
data.urbanrate.iloc[0:] [data.urbanrate.iloc[0:]<50 ] = 0
data.urbanrate.iloc[0:] [data.urbanrate.iloc[0:]>=50 ] = 1

Y = data.BREASTCANCERPER100TH
data.drop(['BREASTCANCERPER100TH'], axis=1, inplace=True)
X = data

clf = tree.DecisionTreeClassifier()

X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size =0.333,random_state =0)
clf.fit(X_train ,Y_train) # clf is a classifier.
Y_pred = clf.fit(X_train , Y_train ).predict(X_test)
print('Test Accuracy: ',accuracy_score(Y_test , Y_pred)) # Test accuracy