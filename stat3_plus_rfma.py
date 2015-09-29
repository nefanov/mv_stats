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
from scipy import optimize
import scipy as sp
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

def MA(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def WMA(x,y,step_size=1,width=1):
    bin_centers  = np.arange(np.min(x),np.max(x)-0.5*step_size,step_size)+0.5*step_size
    bin_avg = np.zeros(len(bin_centers))

    #We're going to weight with a Gaussian function
    def gaussian(x,amp=1,mean=0,sigma=1):
        return amp*np.exp(-(x-mean)**2/(2*sigma**2))

    for index in range(0,len(bin_centers)):
        bin_center = bin_centers[index]
        weights = gaussian(x,mean=bin_center,sigma=width)
        bin_avg[index] = np.average(y,weights=weights)

    return (bin_centers,bin_avg)

def hypRegress(ptp,pir):
    xData = np.arange(len(ptp))
    yData = pir

    xData = np.array(xData, dtype=float)
    yData = np.array(yData, dtype= float)

    def funcHyp(x, qi, exp, di):
        return qi*(1+exp*di*x)**(-1/exp)

    def errfuncHyp(p):
        return funcHyp(xData, p[0], p[1], p[2]) - yData

    #print(xData.min(), xData.max())
    #print(yData.min(), yData.max())

    trialX = np.linspace(xData[0], xData[-1], 1000)

    # Fit an hyperbolic
    popt, pcov = curve_fit(funcHyp, xData, yData)
    print 'popt'
    #print(popt)
    yHYP = funcHyp(trialX, *popt)

    #optimization

    # initial values
    p1, success = sp.optimize.leastsq(errfuncHyp, popt,maxfev=10000)
    print p1

    aaaa = funcHyp(trialX, *p1)

    plt.figure()
    plt.plot(xData, yData, 'r+', label='Data', marker='o')
    plt.plot(trialX, yHYP, 'r-',ls='--', label="Hyp Fit")
    plt.plot(trialX, aaaa, 'y', label = 'Optimized')
    plt.legend()
    plt.show(block=False)
    input('pause')
    return p1

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
Xtrn, Xtest, Ytrn, Ytest = train_test_split(trn, trg, test_size=0.5) #25 % 
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

fig, axes = plt.subplots(ncols=1, figsize=(10,4))
TestModels.R2_Y1.plot( kind='bar', title='R2_Y1')
#TestModels.R2_Y2.plot(ax=axes[1], kind='bar', color='green', title='R2_Y2')
#random_forest is the best ->
model = models[1]
model.fit(Xtrn, Ytrn)
inf=model.feature_importances_
print inf
#then predict with 
print trn
res=model.predict(trn)
print(res)
print trg
print "rmse"
m = metrics.rmse(trg.sure.values,res)

err = metrics.mae(trg.sure.values,res)
print m
print "mean_abs_err"
print err
print "in persentage:"
print(100.0*err/(trg.sure.values.max() - trg.sure.values.min()))

#plot.grid()
#plt.show()

row =  [u'JB', u'p-value', u'skew', u'kurtosis']
jb_test = sm.stats.stattools.jarque_bera(pr)
a = np.vstack([jb_test])
itog = SimpleTable(a, row)
print(itog)

fig = plt.figure()
from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(fig)
ys = t1.dpr.values
xs = t1.speed.values
ax.scatter(xs,ys,t1.sure.values)
plt.show()
f = interpolate.interp2d(xs, ys, t1.sure.values, kind='linear')
xnew=[]
xx=[]
import random
yy=np.arange(100000000,2000000000,100000000)
for i in range(ynew.size):
	xx.append(random.randrange(50,100,10))


znew=f(xx,yy)
print len(znew)
#ax1=Axes3D(fig)
#ax1.plot_surface(xnew,ynew,znew)
#plt.show()

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

f1 = interpolate.interp1d(xf1, yf1,kind='slinear')
xnew1=np.arange(100000000,2000000000,1000000)
ynew1=f1(xnew1)

w = 5
wd = w - 1
dat1 = MA(yf1,w)
dat2 = MA(yf2,w)
dat3 = MA(yf3,w)
print dat1
print "new_val"
print t1.sure.values
i=0
t1['sure']=t1['sure'].astype(float)
ind = t1.index[t1.speed.values==50]
for i in range(len(ind)-wd):
	t1.sure.values[ind[i+wd]] = dat1[i]
i=0
ind = t1.index[t1.speed.values==75]
for i in range(len(ind)-wd):
	t1.sure.values[ind[i+wd]] = dat2[i]
i=0
ind = t1.index[t1.speed.values==100]
for i in range(len(ind)-wd):
	t1.sure.values[ind[i+wd]] = dat3[i]

print "new_val"
print t1.sure.values

ax = Axes3D(fig)
ys = t1.dpr.values
xs = t1.speed.values
#ax.scatter(xs,ys,t1.sure.values)
i=0
znew = []
f2d  = interpolate.interp2d(xs, ys, t1.sure.values, kind='linear')
print f2d(xs[6],ys[6])
for i in range(xs.size):
	znew.append(f2d(xs[i],ys[i]))

#ax.scatter(xs,ys,znew,'ro')
#plt.show()

print "MA results:"
print "rmse:"
m = metrics.rmse(t1.sure.values,znew)
err = metrics.mae(t1.sure.values,znew)
print m
print "mean_abs_err"
print err
print "in persentage:"
print(100.0*err/(t1.sure.values.max() - t1.sure.values.min()))

#= interpolate.interp2d()
'''
f2 = interpolate.interp1d(xf2, yf2,kind='slinear')
xnew2=np.arange(100000000,2000000000,1000000)
ynew2=f2(xnew2)
f3 = interpolate.interp1d(xf3, yf3,kind='slinear')
xnew3=np.arange(100000000,2000000000,1000000)
ynew3=f3(xnew3)

print xf1[4:].size
print len(dat)
plt.plot(xf1, yf1, 'g--', xf1[4:], dat, 'r')#,xf2, yf2, 'b--', xnew2, ynew2, 'r',xf3, yf3, 'b--', xnew3, ynew3, 'r')
#plt.plot(xf1, yf1, 'g--', xreg1, yreg1, 'r',xf2, yf2, 'b--', xnew2, ynew2, 'r',xf3, yf3, 'b--', xnew3, ynew3, 'r')
plt.show()
#hypRegress(xf1,yf1)
#hypRegress(xf2,yf2)
#hypRegress(xf3,yf3)

#popt1, pcov1 = curve_fit(func, xf1, yf1)
#popt2, pcov2 = curve_fit(func, xf2, yf2)
#popt3, pcov3 = curve_fit(func, xf3, yf3)
#plt.figure()
#plt.plot(xf3,yf3,'ko', label="Original Noised data on speed=50")
#plt.plot(xf3,func(xf3,*popt3),'r-',label="log")
#plt.legend
#plt.show()
#######################################################################################

#f = interpolate.interp2d(xs, ys, t1.sure.values, kind='linear')
#ynew=[]
#xnew=[]
#g1 = t1.dpr.values.max()
#g2 = t1.dpr.values.min()

#xnew = np.arange(t1.dpr.values.min(),t1.dpr.values.max(), ( g1-g2 )/10)
#import random
#for i in range(xnew.size):
#	ynew.append(random.randrange(50,100,10))
#chk
#for i in range(t1.index.size):
#	xnew.append(t1.dpr.values[i])
#	ynew.append(t1.speed.values[i])
#znew = f(xnew,ynew)

#print "Znew"
#print(znew.size)
#zz=znew.tolist()

#print xnew.size, ynew.size, znew.size
#ax.scatter
#plt.plot(xnew, znew[0,:])

#plt.show()
'''