# feature importance
os.chdir("C:/DATA/KB_capstone")
names = listdir("C:/DATA/KB_capstone/seperated data") # error point 폴더 안에 데이터만 있어야한다.
names

def csv(name):
    file = pd.read_csv('seperated data/' + name, engine = "python"); file.rename(columns = {"Unnamed: 0"  : "index"}, inplace = True) 
    return file  

Vp = []
for i in names:
    print(i)
    Vp.append(csv(i)) 

indilist = []
for index, name in enumerate(names):
    name = name.replace(".csv", "")
    name = name.replace("상권-", "")
    indilist.append(name)

## df 전처리
Vm = []
for i,df in enumerate(Vp):
    print(i)
    tmp= DataXPro(df)
    Vm.append(tmp) 

## 각 업종별 상관계수 선택. 

coeflist = []

for index, df in enumerate(Vm):
    y = pd.DataFrame(df["THSMON_SELNG_AMT"] / df["STOR_CO"], columns = ["SELNG_PRER_STOR"])
    y = np.sqrt(y)
    s = df.columns.get_loc("THSMON_SELNG_AMT") 
    e = df.columns.get_loc("AGRDE_60_ABOVE_SELNG_CO") 
    X = df.drop(columns = list(df)[s:e+1] ) # 예측의 대상이 되는 매출의 파생지표를 X에서 제거. 
    X = X.iloc[:,5:] # 상권명, 상권번호 등 y의 관련없는 변수 제거.
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

df_coef = coeflist[0]

tablelist = []
for i, df_coef in enumerate(coeflist):
    df_import = df_coef / mean_coef
    table = df_coef.merge(df_import['Value'], how = 'left', on = df_coef.index)
    table = table.rename({'key_0': indilist[i]+' variable_name','Value_x':'A_import','Value_y':'R_import'}, axis = 1)
    tablelist.append(table)        

## save

for i,df in enumerate(tablelist):
    df.to_csv("C:/DATA/KB_capstone/table(Ysqrt_pearson)/" + indilist[i]+".csv",  encoding = "ms949")

## 매출 예측과 유사상권 정의를 위한 중요변수 fitering -> 상관계수의 절대값이 0.2이상인 변수만 사용.

from tqdm import tqdm
## coeff

featurelist = []
for i, df_coef in enumerate(coeflist):
    df_coef = df_coef[df_coef["Value"].abs() > 0.2]
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

for i,df in enumerate(Vsm):
    print(i)
    df = pd.concat([Vm[i][["STDR_YM_CD",'TRDAR_CD']],df], axis = 1)
    y = pd.DataFrame(Vm[i]["THSMON_SELNG_AMT"] / Vm[i]["STOR_CO"], columns = ["SELNG_PRER_STOR"])
    df = pd.concat([df,y], axis = 1)
    df.to_csv('C:/DATA/KB_capstone/filtered_data/'+indilist[i]+'_filterd.csv', encoding = "ms949", index =False)
