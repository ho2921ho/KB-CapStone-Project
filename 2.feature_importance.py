# feature importance
import os
from os import listdir
import pandas as pd
import numpy as np

os.chdir("C:/DATA/KB_capstone")
names = listdir("C:/DATA/KB_capstone/seperated data") # error point 폴더 안에 데이터만 있어야한다.

def csv(name):
    file = pd.read_csv('seperated data/' + name, engine = "python"); file.rename(columns = {"Unnamed: 0"  : "index"}, inplace = True) 
    return file  

# 업종별로 저장한 데이터를 로드하여 Vp에 저장
Vp = []
for i in names:
    print(i)
    Vp.append(csv(i)) 

# indilist에 10가지 업종의 이름을 저장함
indilist = []
for index, name in enumerate(names):
    name = name.replace(".csv", "")
    name = name.replace("상권-", "")
    indilist.append(name)

# 총 유동인구를 단위 면적의 영향을 제하고 비교하기 위해 단위면적당 유동인구수로 변경
def sepfun(df,start,finish,method):
    s = df.columns.get_loc(start)
    e = df.columns.get_loc(finish)
    orgin = df.iloc[:,s:e+1]
    SperO = orgin.apply(lambda x : x/df[method], axis = 0)
    return SperO

def DataXPro(df):
    df = df.fillna(0)
    ## X,y 나누기.
    X = df.copy()
    ## 전처리.. 
    #1. 소비를 상주인구로 나눈다. !!! 소비의 결측값 처리!! 결측은 결측으로!
    s = "EXPNDTR_TOTAMT"
    e = "PLESR_EXPNDTR_TOTAMT"
    tmp= sepfun(df,s,e,"TOT_REPOP_CO")
    tmp[tmp.iloc[:,0] == 0] =  np.nan
    X.iloc[:,range(X.columns.get_loc(s),X.columns.get_loc(e)+1)] = tmp

    # 3. 인구비율 자료를 추가한다. 
    # 3-1 유동인구
    tmp = sepfun(df,"SEX_TOT_FLPOP_CO","FAG_60_ABOVE_SUNTM_6_FLPOP_CO","SEX_TOT_FLPOP_CO")
    tmp.columns = map(lambda x : x.replace('_CO','_RATIO'),list(tmp))
    X = pd.concat([X,tmp], axis = 1) 
    # 3-2 상주인구
    tmp = sepfun(df,"TOT_REPOP_CO","FAG_60_ABOVE_REPOP_CO","TOT_REPOP_CO")
    tmp.columns = map(lambda x : x.replace('_CO','_RATIO'),list(tmp))
    X = pd.concat([X,tmp], axis = 1) 
    # 3-3 직장인구
    tmp = sepfun(df,"TOT_WRC_POPLTN_CO","FAG_60_ABOVE_WRC_POPLTN_CO","TOT_WRC_POPLTN_CO")
    tmp.columns = map(lambda x : x.replace('_CO','_RATIO'),list(tmp))
    X = pd.concat([X,tmp], axis = 1) 
    
    # 2. 단위면적으로 인구자료를 나눈다. 
    ## 2-1 유동인구
    s = X.columns.get_loc("SEX_TOT_FLPOP_CO")
    e = X.columns.get_loc("FAG_60_ABOVE_SUNTM_6_FLPOP_CO")
    X.iloc[:,range(s,e+1)] = sepfun(df,"SEX_TOT_FLPOP_CO","FAG_60_ABOVE_SUNTM_6_FLPOP_CO","Shape_Area")
    ## 2-2 상주인구
    s = X.columns.get_loc("TOT_REPOP_CO")
    e = X.columns.get_loc("FAG_60_ABOVE_REPOP_CO")
    X.iloc[:,range(s,e+1)] = sepfun(df,"TOT_REPOP_CO","FAG_60_ABOVE_REPOP_CO","Shape_Area")
    ## 2-3 직장인구
    s = X.columns.get_loc("TOT_WRC_POPLTN_CO")
    e = X.columns.get_loc("FAG_60_ABOVE_WRC_POPLTN_CO")
    X.iloc[:,range(s,e+1)] = sepfun(df,"TOT_WRC_POPLTN_CO","FAG_60_ABOVE_WRC_POPLTN_CO","Shape_Area")


    #4 소득의 nan 처리.
    s = "MT_AVRG_INCOME_AMT"
    e = "INCOME_SCTN_CD"
    tmp = X.iloc[:,range(X.columns.get_loc(s),X.columns.get_loc(e)+1)]
    tmp[tmp.iloc[:,0] == 0] =  np.nan
    X.iloc[:,range(X.columns.get_loc(s),X.columns.get_loc(e)+1)] = tmp
    
    #5 아파트 변수를 면적으로 나누쟈. 
    s = "APT_HSMP_CO"
    e = "PC_6_HDMIL_ABOVE_HSHLD_CO"
    tmp= sepfun(df,s,e,"Shape_Area")
    X.iloc[:,range(X.columns.get_loc(s),X.columns.get_loc(e)+1)] = tmp
    
    #6. 상권 집객시설을 면적으로 나누자. 
    s = "VIATR_FCLTY_CO"
    e = "BUS_STTN_CO"
    tmp= sepfun(df,s,e,"Shape_Area")
    X.iloc[:,range(X.columns.get_loc(s),X.columns.get_loc(e)+1)] = tmp

    #7. 평균영업개원수 점포수로 나누기. 
    X.loc[:,'AVRG_BSN_MONTH_CO'] = X.loc[:,'AVRG_BSN_MONTH_CO']/X.loc[:,'STOR_CO']
    
    #8. 유사점포수를 상권면적으로 나누기.
    X.loc[:,"SIMILR_INDUTY_STOR_CO"] = X.loc[:,"SIMILR_INDUTY_STOR_CO"] / X.loc[:,"Shape_Area"]
    #9. 다른 업종 점포수를 단위면적으로 나누기. 
    s = "PC방_STOR_CO"
    e = "휴대폰_STOR_CO"
    tmp= sepfun(df,s,e,"Shape_Area")
    X.iloc[:,range(X.columns.get_loc(s),X.columns.get_loc(e)+1)] = tmp
   
    return (X)

## DataXPro에서 정의한 전처리를 요식업의 10가지 업종 데이터에 대해 적용
Vm = []
for i,df in enumerate(Vp):
    print(i)
    tmp= DataXPro(df)
    Vm.append(tmp) 

## 각 업종별 상관계수 선택. 
coeflist = []

def inter2rank(X):
    for i in list(X):
        X.loc[:,i] = X.loc[:,i].rank()

def pear2all(X,y):
    features = X.columns.tolist() #variable name list
    target = list(y)[0] 
    correlations = {}
    for f in features:
        try:
            table = X.loc[:,f].to_frame()
            table['y']= y.copy()
            key = f + ' vs ' + target
            correlations[key] = table.corr().iloc[0,1]
            data_correlations = pd.DataFrame(correlations, index=['Value']).T
            data_correlations.sort_values(ascending = False, axis = 0, na_position = 'last', by = 'Value', inplace = True)
        except:
            print(f)
    return data_correlations

for index, df in enumerate(Vm):
    print(index)
    y = pd.DataFrame(df["THSMON_SELNG_AMT"] / df["STOR_CO"], columns = ["SELNG_PRER_STOR"])
    y = np.sqrt(y)
    s = df.columns.get_loc("THSMON_SELNG_AMT") 
    e = df.columns.get_loc("AGRDE_60_ABOVE_SELNG_CO") 
    X = df.drop(columns = list(df)[s:e+1] ) # 예측의 대상이 되는 매출의 파생지표를 X에서 제거. 
    X = X.iloc[:,5:] # 상권명, 상권번호 등 y의 관련없는 변수 제거.
    inter2rank(X)
    coef = pear2all(X,y)
    coeflist.append(coef)



## 10개 업종의 각 변수와의 평균 coef
sum_coef = coeflist[0].copy()
sum_coef['Value'] = 0 
for df in coeflist:
    sum_coef = sum_coef + df

mean_coef = sum_coef / 10

## 업종마다 상관계수 값이 상위 50개인 변수만을 분석대상으로 하여 
## 매출에 대한 특정 변수의 절대적인 중요도와 상대적인 중요도를 업종별로 정리.

# 기존의 correlation을 절대적 중요도라 칭하며, df_import를 상대적 중요도로 칭하며
# A_import = 절대적 중요도, R_import = 상대적 중요도로 정의

tablelist = []
for i, df_coef in enumerate(coeflist):
    df_import = df_coef - mean_coef ## 상대적 중요도 다시 정으;////
    table = df_coef.merge(df_import['Value'].to_frame(), how = 'left', on = df_coef.index)
    table = table.rename({'key_0': 'variable_name','Value_x':indilist[i],'Value_y':'R_import'}, axis = 1)
    tablelist.append(table)        

os.makedirs('table(Ysqrt_sm_pearson)')

## save
for i,df in enumerate(tablelist):
    df.to_csv("C:/DATA/KB_capstone/table(Ysqrt_sm_pearson)/" + indilist[i]+".csv",  encoding = "ms949")

## 매출 예측과 유사상권 정의를 위한 중요변수 fitering -> 상관계수의 절대값이 0.15이상인 변수만 사용.

## coeff
# df_coef['Value']의 절대값이 0.15이상인 값을 featurelist에 저장
featurelist = []
for i, df_coef in enumerate(coeflist):
    df_coef = df_coef[df_coef["Value"].abs() > 0.15]
    featurelist.append(df_coef.index)

## 
newlist = []
for i in featurelist:
    print(i)
    newlist.append(map(lambda x: x.replace(" vs SELNG_PRER_STOR",""),i))

Vsm = []
for i,df in enumerate(Vm):
    print(i)
    Vsm.append(df.loc[:,newlist[i]])

os.makedirs('filtered_data')

for i,df in enumerate(Vsm):
    print(i)
    df = pd.concat([Vm[i][["STDR_YM_CD",'TRDAR_CD']],df], axis = 1)
    y = pd.DataFrame(Vm[i]["THSMON_SELNG_AMT"] / Vm[i]["STOR_CO"], columns = ["SELNG_PRER_STOR"])
    df = pd.concat([df,y], axis = 1)
    df.to_csv('C:/DATA/KB_capstone/filtered_data/'+indilist[i]+'_filterd.csv', encoding = "ms949", index =False)


# 변수 기준으로 tablelist 재정리.
df0 = tablelist[0] 
df0 = df0.iloc[:,:-1]

for i in range(1,10):
    df = tablelist[i]
    df = df.iloc[:,:-1]
    df0 = df0.merge(df, how = 'left', on = 'variable_name')

os.makedirs('corr_table')
df0.to_csv('C:/DATA/KB_capstone/corr_table/corr_table.csv')
