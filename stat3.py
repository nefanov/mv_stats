import pandas as pd
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.iolib.table import SimpleTable
from sklearn.metrics import r2_score
import ml_metrics as metrics
import datetime,time
import numpy as np
from array import array
def avg(x,y):
	return list(map(lambda a, b: (a + b)/2, x, y))
def dv(x,y):
	return list(map(lambda a, b: a / b, x, y))

dataset = read_csv('dv_1.log',' ',index_col=['time'],error_bad_lines=False)
dataset.head()

t1 = read_csv('time_1_efan_m.log',' ', error_bad_lines=False)
t1.index = dataset.pages_rate.values
prg=t1.sure
prg.head()
int = 0 

#	subset:
pr = prg
res = pr.sort_index()
print "PR"
print res
tmp = [pr.index, pr.values]


pr.index = tmp[:1]
for i in range(pr.values.size):
	pr.values[i] = tmp[1][i]
print pr
itog = pr.describe()
histog = pr.hist()
print itog
print 'V = %f' % (itog['std']/itog['mean'])
X=np.array(dataset.index)
Y=np.array(pr.values)
#print X
#print Y
figure = plt.figure()
plot   = figure.add_subplot(111)
plot.plot(X,Y,'g')
plot.grid()
plt.show()

row =  [u'JB', u'p-value', u'skew', u'kurtosis']
jb_test = sm.stats.stattools.jarque_bera(pr)
a = np.vstack([jb_test])
itog = SimpleTable(a, row)
print itog

print pr
#plt.savefig('spirit.png', format = 'png')
pr.index=dataset.index
print pr
pr = pr[:'2015-09-21 15:16:10']
pr2 =pr
test = sm.tsa.adfuller(pr2)
print 'adf: ', test[0] 
print 'p-value: ', test[1]
print'Critical values: ', test[4]
n = 0
if test[0]> test[4]['5%']:
	print 'stat'
else:
	print 'non-stat'
	print "diff tests"
	while n < 10:
		n = n+1
		pr2 = pr2.diff(periods=4).dropna()
		test = sm.tsa.adfuller(pr2)
		print 'adf: ', test[0]
		print 'p-value: ', test[1]
		print'Critical values: ', test[4]
		if test[0]> test[4]['5%']:
			print 'stat', n
			break
		else:
			print 'non-stat', n

plt.plot(pr2,"r")
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(pr2.values.squeeze(), lags=25, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(pr2, lags=25, ax=ax2)
plt.show()
print pr
print pr.values
src_data_model  = pr[:'2015-09-21 15:16:10']
model = sm.tsa.ARIMA(src_data_model, order=(1,n,1),freq="S").fit(disp=0)	

print model.summary()

q_test = sm.tsa.stattools.acf(model.resid, qstat=True)
print DataFrame({'Q-stat':q_test[1], 'p-value':q_test[2]})

pred = model.predict('2015-09-21 15:16:10','2015-09-22 16:04:53')
trn = pr['2015-09-21 15:16:10':]
print pred.size
#r2 = r2_score(trn, pred)
#print 'R^2: %1.2f' % r2


#metrics.rmse(trn,pred[1:32])

#metrics.mae(trn,pred[1:32])

print pred
print pr2
plt.plot(pr2)

plt.plot(pred[:30],'r--')
plt.show()
