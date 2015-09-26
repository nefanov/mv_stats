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
print t1.index
prg=t1.sure
prg.head()
i = 0
j = 0 
#	subset:
pr = prg
tst100=[]
tst75=[]
tst50=[]
while i in range(pr.values.size):
	for j in range(3):
		if i+j >= pr.values.size:
			break
		tst100.append(pr.values[i+j])
	
	i=i+3
	j = 0
	#if i >= pr.values.size:
	#	break
	for j in range(3):
		if i+j >= pr.values.size:
			break
		tst75.append(pr.values[i+j])
	i=i+3
	j = 0
	
	for j in range(3):
		if i+j >= pr.values.size:
			break
		tst50.append(pr.values[i+j])
	i=i+3
	j = 0
	#if i >= pr.values.size:
	#	break

#tst100 = tst100.sort_index()
print "tst100"
print tst100

pr = pr.sort_index()
print "Data_sorted_by_DPR:"

i = pr.index[0]
regres = []
for i in range( 10000000):
	regres.append(np.polyfit(pr.index, pr.values,i))
	i+=20
print regres

itog = pr.describe()
histog = pr.hist()
print itog
print 'V = %f' % (itog['std']/itog['mean'])
X=np.array(pr.index)
Y=np.array(regres)
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


#r2 = r2_score(trn, pred)
#print 'R^2: %1.2f' % r2


#metrics.rmse(trn,pred[1:32])

#metrics.mae(trn,pred[1:32])

