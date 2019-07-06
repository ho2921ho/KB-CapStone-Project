#%% raw 데이터 불러오기.
import os
from os import listdir
import pandas as pd
import collections
from numpy import isnan

os.chdir("C:/DATA/KB_capstone")
names = listdir("C:/DATA/KB_capstone/data") # error point 폴더 안에 데이터만 있어야한다.
names

def csv(name):
    file = pd.read_csv('data/' + name, engine = "python"); file.rename(columns = {"Unnamed: 0"  : "index"}, inplace = True) 
    return file  

V = []
for i in names:
    print(i)
    V.append(csv(i))     

#%% 사용할 기간만 추출 201705 ~ 201804
Vfilterd = []
for i in range(len(V)):
    print(i)
    temp = V[i][(V[i].iloc[:,0] >=  201705) & (V[i].iloc[:,0] <=  201804)]
    Vfilterd.append(temp)

#%% 개별 데이터 셋 변수선언
table_names = ("ihpopf","iincomf","iaptf","istrf","icpopf","ifacf","isalef","impopf")

dd = collections.OrderedDict(zip(table_names,Vfilterd))
globals().update(dd)

#%% 점포수&매출 업종별 filter and merge
indilist = ['양식집','치킨집','제과점','분식집','일식집','커피음료','중국집','호프간이주점','한식음식점','패스트푸드점']
indilist = sorted(indilist)

sepistr = sepfilter(istrf,indilist)
sepisale = sepfilter(isalef,indilist)

month = pd.DataFrame(sorted(list(set(ihpopf['STDR_YM_CD'])), reverse= True), columns = ["STDR_YM_CD"])
month["tmp"] = 1 

clust_iX = pd.DataFrame({"TRDAR_CD":list(range(1,1745))})
clust_iX["tmp"] = 1
clust_iX =  pd.merge(month,clust_iX, on = ["tmp"])
del clust_iX["tmp"] 

for i in [0,1,2,4,5,7]:
    df = Vfilterd
    df[i].drop("TRDAR_CD_NM", axis = 1)
    
for i in [0,1,2,4,5,7]:
    df = Vfilterd
    clust_iX = pd.merge(clust_iX,df[i], how='left', on=['STDR_YM_CD','TRDAR_CD']) 

shape = pd.read_csv("shape(from_Qgis).csv", engine = 'python')
shape.rename(columns ={'ALLEY_TRDA':'TRDAR_CD'}, inplace = True)
shape = shape.loc[:,['TRDAR_CD','Shape_Area']]

clust_iX = pd.merge(clust_iX,shape, how = 'left', on = ['TRDAR_CD'])

#%% 업동간 집적효과 항 만들기 ex) 중식집 vs 당구장의 관계를 파악할 수 있도록하기 위함. 
indilist = sorted(list(set(istrf['SVC_INDUTY_CD_NM'])))
indilist.remove('노인요양시설')

sepistrall = sepfilter(isalef,indilist)

allstrlist = []
for df in sepistrall:
    colname = sorted(list(set(df['SVC_INDUTY_CD_NM'])))[0]
    print(colname)
    tmp = df.loc[:,["STDR_YM_CD","TRDAR_CD",'STOR_CO']]
    tmp.rename(columns ={'STOR_CO':colname+'_STOR_CO'}, inplace = True)
    allstrlist.append(tmp)

allstr = clust_iX.copy()
for df in allstrlist:
    allstr = pd.merge(allstr,df, how = 'left', on = ["STDR_YM_CD","TRDAR_CD"])

list(allstr)
s = allstr.columns.get_loc('PC방_STOR_CO')
e = allstr.columns.get_loc('휴대폰_STOR_CO')
allstr = allstr.iloc[:,range(s,e+1)]
allstr[isnan(allstr)] = 0

clust_iX = pd.concat([clust_iX,allstr], axis = 1)

## 점포 집적효과 항 만들기 2단계, 점포에 있는 점포수...
indilist = sorted(list(set(istrf['SVC_INDUTY_CD_NM'])))
indilist.remove('노인요양시설')
sepistrall2 = sepfilter(istrf,indilist)

allstrlist2 = []
for df in sepistrall2:
    colname = sorted(list(set(df['SVC_INDUTY_CD_NM'])))[0]
    print(colname)
    tmp = df.loc[:,["STDR_YM_CD","TRDAR_CD",'STOR_CO']]
    tmp.rename(columns ={'STOR_CO':colname+'_점포수의_'+'_STOR_CO'}, inplace = True)
    allstrlist2.append(tmp)

allstr2 = clust_iX.copy()
for df in allstrlist2:
    allstr2 = pd.merge(allstr2,df, how = 'left', on = ["STDR_YM_CD","TRDAR_CD"])

list(allstr2)
s = allstr2.columns.get_loc('PC방_점포수의__STOR_CO')
e = allstr2.columns.get_loc('휴대폰_점포수의__STOR_CO')
allstr2 = allstr2.iloc[:,range(s,e+1)]
allstr2[isnan(allstr2)] = 0

clust_iX = pd.concat([clust_iX,allstr2], axis = 1)
clust_iX .to_csv("pre_model data_X")


#%% X&점포수&매출 최종 merge 후 저장.
for i in range(len(sepistr)): # 불필요한 열 제거.
    del sepistr[i]["TRDAR_CD_NM"];del sepistr[i]["SVC_INDUTY_CD"]

Trdarlist = []
for i in range(len(sepistr)):    
    Trdar = pd.merge(sepisale[i],sepistr[i], how='left', on=['STDR_YM_CD','TRDAR_CD']) 
    Trdar = pd.merge(Trdar,clust_iX, how='left', on=['STDR_YM_CD','TRDAR_CD'])
    Trdarlist.append(Trdar)


for index, df in enumerate(Trdarlist):
    df.rename(columns={"SVC_INDUTY_CD_NM_x":"SVC_INDUTY_CD_NM",'STOR_CO_x':'STOR_CO'}, inplace = True)
    del df['STOR_CO_y']; del df['SVC_INDUTY_CD_NM_y']   
    Trdarlist[index] = df

#### for csv.
indilist = sorted(['양식집','치킨집','제과점','분식집','일식집','커피음료','중국집','호프간이주점','한식음식점','패스트푸드점'])

for i in range(len(Trdarlist)):
    print(i)
    Trdarlist[i].to_csv("C:/DATA/KB_capstone/seperated data" +"/상권-"+indilist[i]+".csv", encoding = "ms949", index = False)
    



