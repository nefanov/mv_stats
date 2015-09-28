import pandas as pd
import os
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.iolib.table import SimpleTable
from sklearn.metrics import r2_score
import ml_metrics as metrics
import datetime,time
import numpy as np
from math import log, exp
from array import array
from scipy import interpolate
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm 

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def avg(x,y):
	return list(map(lambda a, b: (a + b)/2, x, y))
def dv(x,y):
	return list(map(lambda a, b: a / b, x, y))



dataset = read_csv('dv_1.log',' ',index_col=['time'],error_bad_lines=False)

dataset.head()
dataset.index = pd.to_datetime((dataset.index*1e9).astype(int))
#print dataset

#dataset.to_csv("1.txt")

#dataset = read_csv('dv_0.log',' ',index_col=['time'],error_bad_lines=False)
#dataset.head()
#dataset.index = pd.to_datetime((dataset.index*1e9).astype(int))

#print dataset
#f = open("0.txt","w")
#f.write(dataset.index)#+" "+dataset.values)

t1 = read_csv('time_1_efan_m.log',' ', error_bad_lines=False)
t1.index = dataset.pages_rate.values

prg=t1.sure
prg.head()
print("data")
print(prg)
i = 0
j = 0 
#	subset:
pr = prg
speed=[]

tst100=[]
tst75=[]
tst50=[]
while i in range(pr.values.size):
	for j in range(6):
		if i+j >= pr.values.size:
			break
		#tst100.append([pr.index[i+j], pr.values[i+j]])
		speed.append(100)
	
	i=i+6
	j = 0
	
	for j in range(6):
		if i+j >= pr.values.size:
			break
		#tst75.append([pr.index[i+j], pr.values[i+j]])
		speed.append(75)
	i=i+6
	j = 0
	
	for j in range(6):
		if i+j >= pr.values.size:
			break
		#st50.append([pr.index[i+j], pr.values[i+j]])
		speed.append(50)
	i=i+6
	j = 0
	
t1.insert(0,'speed',speed)
t1.sort_index(ascending=True, inplace=True)

i =0 

#then we can use any column independently


#tst100 = tst100.sort_index()
#print "tst100"
#print tst100

pr = pr.sort_index()

#i = pr.index[0]
#regres = []
#for i in range( 10000000):#
#	regres.append(np.polyfit(pr.index, pr.values,i))
#	i+=20
#print regres
#
regres = pr.interpolate()
f = interpolate.interp1d(pr.index, pr.values)
ynew = f(pr.index)
print(ynew)
itog = pr.describe()
histog = pr.hist()
print(itog)
print('V = %f' % (itog['std']/itog['mean']))
X=np.array(pr.index)
Y=pr.values
figure = plt.figure()

plot   = figure.add_subplot(111)
plot.plot(X,Y,'g')
#plot.plot(X,ynew,'r--')
print("Corr_table")
ct = t1.corr()
print(ct)

print('DDDDD')
print(t1)

#init target:
inew = np.arange(0,t1.index.size, 1)
t1.insert(1,'dpr',t1.index)
t1.index = inew

trg = t1[['sure']]
trn = t1.drop(['end','sure'],axis=1)
print(trg)
#init set of models:
models = [LinearRegression(), RandomForestRegressor(n_estimators=100, max_features ='sqrt'),KNeighborsRegressor(n_neighbors=6),LogisticRegression()]
#init subset
Xtrn, Xtest, Ytrn, Ytest = train_test_split(trn, trg, test_size=0.4)

TestModels = DataFrame()
tmp = {}

for model in models:

	m = str(model)
	tmp['Model'] = m[:m.index('(')]    
	i=0
	model.fit(Xtrn, Ytrn.sure) 
	tmp['R2_Y%s'%str(i+1)] = r2_score(Ytest.sure, model.predict(Xtest))
	TestModels = TestModels.append([tmp])

TestModels.set_index('Model', inplace=True)

fig, axes = plt.subplots(ncols=2, figsize=(10,4))
TestModels.R2_Y1.plot(ax=axes[0], kind='bar', title='R2_Y1')
#TestModels.R2_Y2.plot(ax=axes[1], kind='bar', color='green', title='R2_Y2')

model = models[1]
model.fit(Xtrn, Ytrn)
inf=model.feature_importances_
print inf

plot.grid()
plt.show()

row =  [u'JB', u'p-value', u'skew', u'kurtosis']
jb_test = sm.stats.stattools.jarque_bera(pr)
a = np.vstack([jb_test])
itog = SimpleTable(a, row)
print(itog)

fig = plt.figure()
from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(fig)
ys = t1.dpr
xs = t1.speed.values
ax.scatter(xs,ys,t1.sure.values)
plt.show()

#######################################################################################
tf1=t1[t1.speed==50]
xf1=tf1.dpr.values
yf1=tf1.sure.values
tf1.to_csv("out50")
#add plot
tf2=t1[t1.speed==75]
xf2=tf2.dpr.values
yf2=tf2.sure.values
tf2.to_csv("out75")
#add plot
tf3=t1[t1.speed==100]
xf3=tf3.dpr.values
yf3=tf3.sure.values
tf3.to_csv("out100")#add plot
popt1, pcov1 = curve_fit(func, xf1, yf1)
popt2, pcov2 = curve_fit(func, xf2, yf2)
popt3, pcov3 = curve_fit(func, xf3, yf3)
#plt.figure()
#plt.plot(xf3,yf3,'ko', label="Original Noised data on speed=50")
#plt.plot(xf3,func(xf3,*popt3),'r-',label="log")
#plt.legend
#plt.show()
#######################################################################################

f = interpolate.interp2d(xs, ys, t1.sure.values, kind='linear')
ynew=[]
xnew=[]
g1 = t1.dpr.values.max()
g2 = t1.dpr.values.min()

#xnew = np.arange(t1.dpr.values.min(),t1.dpr.values.max(), ( g1-g2 )/10)
#import random
#for i in range(xnew.size):
#	ynew.append(random.randrange(50,100,10))
#chk
for i in range(t1.index.size):
	xnew.append(t1.dpr.values[i])
	ynew.append(t1.speed.values[i])
znew = f(xnew,ynew)

print "Znew"
print(znew.size)
zz=znew.tolist()

#print xnew.size, ynew.size, znew.size
#ax.scatter
plt.plot(xnew, znew[0,:])

plt.show()