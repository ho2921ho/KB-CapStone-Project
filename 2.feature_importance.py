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

for index, df in tqdm(enumerate(Vm)):
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

tablelist = []
for i, df_coef in enumerate(coeflist):
    df_coef = coeflist[0]
    df_import = df_coef / mean_coef
    row = 50
    df_coef.sort_values(ascending = False, axis = 0, by = 'Value', inplace = True)
    it = df_coef.iloc[row-1,0]
    df_coef = df_coef[df_coef.iloc[:,0] >= it]
    
    df_import.sort_values(ascending = False, axis = 0, by = 'Value', inplace = True)
    top10_Relative_import = df_import.iloc[0:row,:]
    top10_Relative_import.columns = [indilist[i] + 'Rimport']
    top10_Absolute_import = df_coef.iloc[0:row,:]
    top10_Absolute_import.columns = [indilist[i] + 'Aimport']
    table = pd.concat([top10_Absolute_import,top10_Relative_import], axis = 1)
    table.sort_values(ascending = False, axis = 0, by = indilist[i] + 'Aimport', inplace = True)
    tablelist.append(table)        

tmp = tablelist[1]    
for table in tablelist:
    print(table)
    

os.chdir("C:/DATA/KB_capstone")

# s
for i,df in enumerate(tablelist):
    df.to_excel("table(Ysqrt_pearson)"/ + indilist[i]+".xlsx",  encoding = "ms949")

'''
#################### 상상위 10개 찾기..
os.chdir("C:/Users/renz/Documents/GitHub/KB-CapStone-Project")
names = listdir("C:/Users/renz/Documents/GitHub/KB-CapStone-Project/table(Ysqrt)") # error point 폴더 안에 데이터만 있어야한다.
names

def csv(name):
    file = pd.read_excel('table(Ysqrt)/' + name); file.rename(columns = {"Unnamed: 0"  : "index"}, inplace = True) 
    return file  

Vi = []
for i in names:
    print(i)
    Vi.append(csv(i)) 
    
for df in Vi:
    A = df[df.iloc[:,0] > 0.2]
    R = df[df.iloc[:,1] > 1.2]
    ARimport = []
    for index in list(A.index):
        if index in list(R.index):
            ARimport.append(index)
    print(A.ix[list(ARimport)])

####################### 과밀 feature 만들기...-> 유사점포수?
#### 업종 y 들 간의 상관관계

month = pd.DataFrame(sorted(list(set(Vm[0]['STDR_YM_CD'])), reverse= True), columns = ["STDR_YM_CD"])
month["tmp"] = 1 

key = pd.DataFrame({"TRDAR_CD":list(range(1,1745))})
key["tmp"] = 1
key =  pd.merge(month,key, on = ["tmp"])
del key["tmp"] 

Ydf = key.copy()
for i, df in enumerate(Vm):
    print(i)
    y = pd.DataFrame(df["THSMON_SELNG_AMT"] / df["STOR_CO"], columns = [indilist[i]])
    y["STDR_YM_CD"] = df["STDR_YM_CD"]
    y["TRDAR_CD"] = df["TRDAR_CD"]
    Ydf = pd.merge(Ydf,y,how = 'left', on = ["STDR_YM_CD","TRDAR_CD"])
    
Ycorr = Ydf.corr()
'''
###################### 중요변수 fitering -> 상관계수의 절대값이 1.2이상인 곳만! 

from tqdm import tqdm
## coeff



featurelist = []
for i, df_coef in enumerate(coeflist):
    df_coef = df_coef[df_coef["Value"].abs() > 1.2]
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
    df.to_csv('C:/Users/renz/Documents/GitHub/KB-CapStone-Project/filterd data(00.0)/'+indilist[i]+'_filterd.csv', encoding = "ms949", index =False)


test = X.corr()
test2= test.ix["MT_AVRG_INCOME_AMT"]
