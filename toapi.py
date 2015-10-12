import os
from scipy import interpolate
from pandas import read_csv, DataFrame
from sklearn.ensemble import RandomForestRegressor
#from sklearn.cross_validation import train_test_split
from scipy.optimize import curve_fit

def MA(a, n=3):
    rev = a[:n]
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def rf_train(data,n_estimators=100):
	model=RandomForestRegressor(n_estimators,max_features ='sqrt')
	Ytrn = data['mig_time']
	Xtrn = data.drop(['mig_time'],axis=1)
	model.fit(Xtrn,Ytrn)

	return model

def rf_test(Xtst,model):
	return model.predict(Xtst)

def gen_interp(data,wd=5):
	MA_res=[]
	for i in range(data.speed.values.min(),data.speed.values.max(),10):
		MA_res = MA(data.mig_time.values,wd)
		ind = data.index[data.speed.values==i]
		for j in range(len(ind)-wd):
			data.mig_time.values[ind[j+wd]] = Ma_res[j]
	f = interpolate.interp2d(data.dpr.values,data.speed.values,data.mig_time.values, kind='linear')
	return f

def use_interp(data,f):
	x=data.dpr.values
	y=data.speed.values
	return f(x,y)

def parse_csv(fn='1.log')
	data=read_csv(fn,' ',error_bad_lines=False)
	return data

# 							*****How to use**** 
#This comment describes model's usage because of different ways to evaluate them
#
# 1. Read train data from csv file using data=parse_csv(filename)
# Data is represented in DataFrame object because of customization, optimization and compact code
# field's heads were hardcoded, that's why structure of csv is:
# |_index_|_dpr_|_speed_|_mig_time_| 
#
# 2. If you use RandomForestRegressor:
#	a) train your model on 'data' using model=rf_train(data) (once before runtime)
#	b) evaluate arbitrary point or group of points using res=rf_test(Xtst,model),
#	where Xtst - reduced data in format: 
#	|_index_|_dpr_|_speed_|. Output format is : |_index_|_mig_time_|

#3. If you use Interpolation:
#	a) Construct interpolation: f=gen_interp(data,wd=5) (once before runtime)
#	b) evaluate arbitrary point or group of points using res=use_interp(data,f), 
#   function returns a list of approximated mig_time values

#4. Any questions?  
#	nefanov90@gmail.com
#   github.com/nefanov/mv_stat
