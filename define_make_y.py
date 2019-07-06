os.chdir("C:/Users/renz/Documents/GitHub/KB-CapStone-Project/seoul market")
osale = pd.read_csv('data/상권배후지-추정매출.csv',engine = "python")
isale = pd.read_csv('data/상권-추정매출.csv',engine = "python")
istr = pd.read_csv("data/상권-점포.csv", engine = 'python')
ostr = pd.read_csv("data/상권배후지-점포.csv", engine = 'python')
gc.collect()
osale.columns = list(isale)
ostr.columns = list(ostr)
# detection for industry selection 
isale_pivot = isale.pivot_table(values = 'TRDAR_CD', index = ['STDR_YM_CD'], 
                                columns = 'SVC_INDUTY_CD_NM',aggfunc= 'count')
osale_pivot = osale.pivot_table(values = 'TRDAR_CD', index = ['STDR_YM_CD'], 
                                columns = 'SVC_INDUTY_CD_NM',aggfunc= 'count')

# make y about sale Ysale function

Yisale = Ysale(isale)
Yosale = Ysale(osale)

Yisale.loc[:,['STDR_YM_CD','SVC_INDUTY_CD_NM','TRDAR_STR_SELNG_AMT']].groupby(['STDR_YM_CD','SVC_INDUTY_CD_NM']).hist()

# make y about 폐업률.... 자료가 너무 sparse 하다. .한 골목상권에서 폐업한 점포가 한달에 한두개정도..?
list(istr)
Yistr = istr.loc[:,['STDR_YM_CD','TRDAR_CD','SVC_INDUTY_CD_NM','CLSBIZ_RT']]
tmp = Yistr.groupby(['STDR_YM_CD','SVC_INDUTY_CD_NM'], as_index = False).mean()
tmp.columns = ['STDR_YM_CD', 'SVC_INDUTY_CD_NM','TRDAR_CD_f',"THSMON_AVG_CLSVIZ_RT"]
Yistr = pd.merge(Yistr, tmp, how = 'left', on = ['STDR_YM_CD','SVC_INDUTY_CD_NM'])
del Yistr['TRDAR_CD_f']

# make y about 평균영업개월수 새로지어진 대라서 그럴 수 도 있구...
list(isale)
Yibsn = isale.loc[:,['STDR_YM_CD','TRDAR_CD','SVC_INDUTY_CD_NM','AVRG_BSN_MONTH_CO']]
tmp = Yibsn.groupby(['STDR_YM_CD','SVC_INDUTY_CD_NM'], as_index = False).mean()
tmp.columns = ['STDR_YM_CD', 'SVC_INDUTY_CD_NM','TRDAR_CD_f',"THSMON_AVRG_BSN_MONTH_CO'"]
Yibsn = pd.merge(Yibsn, tmp, how = 'left', on = ['STDR_YM_CD','SVC_INDUTY_CD_NM'])
del Yibsn['TRDAR_CD_f']

## 폐업률과 매출액의 상관관계. -0.2524.
list(test)
test = pd.merge(Yisale, Yistr, how = 'left', on = ['STDR_YM_CD','TRDAR_CD','SVC_INDUTY_CD_NM'])
test = test[-(isnan(test.loc[:,'THSMON_AVG_CLSVIZ_RT']))]
plt.scatter(test['THSMON_STR_SELNG_AMT'],test['THSMON_AVG_CLSVIZ_RT'])
plt.show()
pearsonr(test['THSMON_STR_SELNG_AMT'],test['THSMON_AVG_CLSVIZ_RT'])

## 면적과 매출액의 상관관계. (201804기준)
test = Yisale[Yisale["STDR_YM_CD"] == 201804].groupby("TRDAR_CD").mean()

## 관측 점포수 2017.11
sum(isale[isale["STDR_YM_CD"] == 201711].iloc[:,-1])
shape = pd.read_csv("tdr_shape.csv", engine = 'python')
list(shape)
sum(np.array(shape.loc[:,['Shape_Area']]))
tmp = shape.pivot_table(values = 'ALLEY_TRDA', index = ['SIGNGU_CD'], columns = 'STDR_YM_CD',aggfunc= 'count')

