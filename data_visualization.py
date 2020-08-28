import pandas as pd 
import matplotlib.pyplot as plt
import os
import glob 
import csv
from scipy.stats import ks_2samp
import scipy.stats as stats

# scratch directory contains performance files of queries method
files = glob.glob("./scratch/*.csv")
print(files)
'''
result file form:
week, start_dat, end_day, prec, rec, rev
'''

prec = []
rec =[]
rev =[]
# print(prec)

for f in files:
	cur = pd.read_csv(f,header=None)
	prec.append(cur.iloc[:,3])
	rec.append(cur.iloc[:,4])
	rev.append(cur.iloc[:,5])

df_prec = prec[0]
df_rec = rec[0]
df_rev = rev[0]
# create dataframe for each pre, rec, rev
for i in range(1,len(prec)):
	df_prec = pd.concat([df_prec,prec[i]], axis =1)
	df_rec = pd.concat([df_rec,rec[i]], axis =1)
	df_rev = pd.concat([df_rev,rev[i]], axis =1)

# assign colum name in order in folder
query_strategies = ['badge_DATE','hybrid','diversity','badge','DATE','random']
df_prec.columns = query_strategies
df_rec.columns = query_strategies
df_rev.columns = query_strategies

x_axis = df_prec.index

def plot_data(df,option):
	fig = plt.figure()
	ax = plt.axes()
	for q in query_strategies:
		y_axis = df.loc[:,q]
		y_axis = y_axis.T
		plt.plot(x_axis,y_axis, label = q)
		plt.xlabel("week")
		plt.ylabel(option)
		plt.legend()
	plt.show()

plot_data(df_prec,'precision')
plot_data(df_rec,'recall')
plot_data(df_rev,'revenue')

# def ANOVA(df1, df2):
# 	F, p = stats.f_oneway(df1,df2)
# 	print('F-Statistic=%.3f, p=%.3f' % (F, p))
# 	return F, p

# F, p = ANOVA(df_rev.loc[:,'badge'],df_rev.loc[:,'DATE'])
# if p < 0.05:
# 	print("Reject Null hypothesis - no differences between these 2")
# print("========================================================")
# df_s = [df_prec,df_rec,df_rev]

# for d in df_s:
# 	# print("-----mean")
# 	# print(d.mean())
# 	# print("-----var")
# 	# print(d.var())
# 	with open("[scratch]mean&var.csv","a") as f:
# 		wr = csv.writer(f, delimiter = ',')
# 		wr.writerow(d.mean())
# 		wr.writerow(d.var())



