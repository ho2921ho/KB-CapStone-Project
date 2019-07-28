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

# 지정한 path내에 존재하는 파일을 V 리스트내에 기존 파일명 그대로 데이터프레임으로 불러 저장함
V = []
for i in names:
    print(i)
    V.append(csv(i))     

#%% 사용할 기간만 추출 201705 ~ 201804
# 모든 데이터가 담긴 리스트를 순환하며 기간에 맞는 데이터만을 추출하여 Vfilterd 리스트에 저장
Vfilterd = []
for i in range(len(V)):
    print(i)
    temp = V[i][(V[i].iloc[:,0] >=  201705) & (V[i].iloc[:,0] <=  201804)]
    Vfilterd.append(temp)

#%% 개별 데이터 셋 변수선언
# 상권주거인구/상권소득/상권아파트/상권점포수/상권직장인구/상권내시설/상권내매출정보/상권내유동인구
table_names = ("ihpopf","iincomf","iaptf","istrf","icpopf","ifacf","isalef","impopf")

dd = collections.OrderedDict(zip(table_names,Vfilterd))
globals().update(dd)


#%% 점포수&매출 업종별 filter and merge
# 요식업으로 컨버전스를 좁히기로 하였기에, 요식업에 해당하는 데이터만을 추출
indilist = ['양식집','치킨집','제과점','분식집','일식집','커피음료','중국집','호프간이주점','한식음식점','패스트푸드점']
indilist = sorted(indilist)

# indilist에 넣은 업종만을 필터링해주는 함수
def sepfilter (df,indilist):
    seplist = []
    for i in indilist:
        tmp = df[df.iloc[:,4]== i]
        seplist.append(tmp)
    return seplist

sepistr = sepfilter(istrf,indilist)
sepisale = sepfilter(isalef,indilist)

# 데이터의 Date를 Unique한 YYYYMM을 확인, 저장
month = pd.DataFrame(sorted(list(set(ihpopf['STDR_YM_CD'])), reverse= True), columns = ["STDR_YM_CD"])
month["tmp"] = 1 

# 1번부터 1744번 상권까지 번호를 clust_iX에 저장함
clust_iX = pd.DataFrame({"TRDAR_CD":list(range(1,1745))})
clust_iX["tmp"] = 1

# tmp를 key로서 년월과 상권코드를 merge한 후, tmp를 제거
clust_iX =  pd.merge(month,clust_iX, on = ["tmp"])
del clust_iX["tmp"] 

# 현 분석에서 최종적으로 사용할 데이터에 대해 df에 저장
for i in [0,1,2,4,5,7]:
    df = Vfilterd
    df[i].drop("TRDAR_CD_NM", axis = 1)

# 최종적으로 사용할 데이터 중, clust_iX의 기간, 상권을 key로 merge
for i in [0,1,2,4,5,7]:
    df = Vfilterd
    clust_iX = pd.merge(clust_iX,df[i], how='left', on=['STDR_YM_CD','TRDAR_CD']) 

# Qgis를 통해 상권에 따른 면적정보가 담긴 shp파일을 csv로 encoding한 파일을 불러옴
shape = pd.read_csv("shape(from_Qgis).csv", engine = 'python')
shape.rename(columns ={'ALLEY_TRDA':'TRDAR_CD'}, inplace = True)
shape = shape.loc[:,['TRDAR_CD','Shape_Area']]

# 상권번호를 key로 상권별 면적 정보가 담긴 shape를 clust_iX와 merge
clust_iX = pd.merge(clust_iX,shape, how = 'left', on = ['TRDAR_CD'])

#%% 업동간 집적효과 항 만들기 ex) 중식집 vs 당구장의 관계를 파악할 수 있도록하기 위함
# 전체 업종에서, 노인요양시설을 제외한 모든 unique한 업종을 indilist에 저장함 
indilist = sorted(list(set(istrf['SVC_INDUTY_CD_NM'])))
indilist.remove('노인요양시설')

# 노인요양시설을 제외한 전체 업종에 대해 매출정보 데이터와 merge
sepistrall = sepfilter(isalef,indilist)

# sepistrall 내의 업종에 대해, 년월별, 상권별 업종별 점포수를 체크함
allstrlist = []
for df in sepistrall:
    colname = sorted(list(set(df['SVC_INDUTY_CD_NM'])))[0]
    print(colname)
    tmp = df.loc[:,["STDR_YM_CD","TRDAR_CD",'STOR_CO']]
    tmp.rename(columns ={'STOR_CO':colname+'_STOR_CO'}, inplace = True)
    allstrlist.append(tmp)

# clust_iX를 복사하여 allstr로 저장한 후, allstrlist에 저장한 업종별 점포수를 all_str에 merge
allstr = clust_iX.copy()
for df in allstrlist:
    allstr = pd.merge(allstr,df, how = 'left', on = ["STDR_YM_CD","TRDAR_CD"])

# allstr의 columns을 확인 후, 점포별 개수의 시작 변수인 PC방의 점포수와 마지막 변수인 휴대폰의 점포수 columns의 index 넘버를 체크
list(allstr)
s = allstr.columns.get_loc('PC방_STOR_CO')
e = allstr.columns.get_loc('휴대폰_STOR_CO')
# 전 업종의 점포수 데이터만을 추출 후 nan값은 점포가 존재하지 않기에 0값으로 대체
allstr = allstr.iloc[:,range(s,e+1)]
allstr[isnan(allstr)] = 0

# nan값을 0값으로 대체한 데이터를 clust_iX와 merge
clust_iX = pd.concat([clust_iX,allstr], axis = 1)

## 점포 집적효과 항 만들기 2단계, 점포에 있는 점포수
# 노인요양시설을 제외한 전 업종을 점포 수 데이터에서 필터하여 sepistrall2에 저장
indilist = sorted(list(set(istrf['SVC_INDUTY_CD_NM'])))
indilist.remove('노인요양시설')
sepistrall2 = sepfilter(istrf,indilist)

# 점포 수 데이터에서 년월별, 상권별에 따른 업종별 점포수를 allstrlist2에 저장함
allstrlist2 = []
for df in sepistrall2:
    colname = sorted(list(set(df['SVC_INDUTY_CD_NM'])))[0]
    print(colname)
    tmp = df.loc[:,["STDR_YM_CD","TRDAR_CD",'STOR_CO']]
    tmp.rename(columns ={'STOR_CO':colname+'_점포수의_'+'_STOR_CO'}, inplace = True)
    allstrlist2.append(tmp)

# clust_iX를 copy한 allstr2와 df를 merge
allstr2 = clust_iX.copy()
for df in allstrlist2:
    allstr2 = pd.merge(allstr2,df, how = 'left', on = ["STDR_YM_CD","TRDAR_CD"])

# 점포수 column의 시작인 PC방 점포수와 마지막인 휴대폰 점포수의 위치를 체크 후, 해당 데이터만을 빼내어 nan값을 0으로 줌
list(allstr2)
s = allstr2.columns.get_loc('PC방_점포수의__STOR_CO')
e = allstr2.columns.get_loc('휴대폰_점포수의__STOR_CO')
allstr2 = allstr2.iloc[:,range(s,e+1)]
allstr2[isnan(allstr2)] = 0

# 위와 같이 처리된 값을, clust_iX에 합친 후, csv로 저장함
clust_iX = pd.concat([clust_iX,allstr2], axis = 1)

clust_iX.to_csv("pre_model data_X.csv", encoding = 'utf-8')

#%% X&점포수&매출 최종 merge 후 저장.
for i in range(len(sepistr)): # 불필요한 열 제거.
    del sepistr[i]["TRDAR_CD_NM"];del sepistr[i]["SVC_INDUTY_CD"]

# 요식업에 해당하는 업종에 대해, 매출과 점포수 데이터를 merge하여 Trdarlist에 저장
Trdarlist = []
for i in range(len(sepistr)):    
    Trdar = pd.merge(sepisale[i],sepistr[i], how='left', on=['STDR_YM_CD','TRDAR_CD']) 
    Trdar = pd.merge(Trdar,clust_iX, how='left', on=['STDR_YM_CD','TRDAR_CD'])
    Trdarlist.append(Trdar)

# merge시 데이터 간 중복으로 인해 x, y가 붙는 변수에 대해 x는 원래 변수명으로 바꾸며 y는 삭제
for index, df in enumerate(Trdarlist):
    df.rename(columns={"SVC_INDUTY_CD_NM_x":"SVC_INDUTY_CD_NM",'STOR_CO_x':'STOR_CO'}, inplace = True)
    del df['STOR_CO_y']; del df['SVC_INDUTY_CD_NM_y']   
    Trdarlist[index] = df

#### for csv.

indilist = sorted(['양식집','치킨집','제과점','분식집','일식집','커피음료','중국집','호프간이주점','한식음식점','패스트푸드점'])

# 업종별로 데이터를 추출 후, 별개의 csv파일로서 저장함
for i in range(len(Trdarlist)):
    print(i)
    Trdarlist[i].to_csv("C:/DATA/KB_capstone/seperated data" +"/상권-"+indilist[i]+".csv", encoding = "ms949", index = False)
    



