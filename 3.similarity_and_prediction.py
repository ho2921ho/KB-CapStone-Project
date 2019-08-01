##3.1 유사상권 정의와 매출 예측을 위한 model data 만들기. 

os.chdir("C:/DATA/KB_capstone")
names = listdir("C:/DATA/KB_capstone/filtered_data") # error point 폴더 안에 데이터만 있어야한다.
names

def csv(name):
    file = pd.read_csv('filtered_data/' + name, engine = "python"); file.rename(columns = {"Unnamed: 0"  : "index"}, inplace = True) 
    return file  

Vf = []
for i in names:
    print(i)
    Vf.append(csv(i)) 

# 모델링을 위한 전처리 방식을 담은 DataXPro_for_model을 모델링 하기 이전에 적용함
def DataXPro_for_model(df):
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

    #7. 다른 업종 점포수를 단위면적으로 나누기. 
    s = "PC방_STOR_CO"
    e = "휴대폰_STOR_CO"
    tmp= sepfun(df,s,e,"Shape_Area")
    X.iloc[:,range(X.columns.get_loc(s),X.columns.get_loc(e)+1)] = tmp
   
    return (X)
    
def sepfun(df,start,finish,method):
    s = df.columns.get_loc(start)
    e = df.columns.get_loc(finish)
    orgin = df.iloc[:,s:e+1]
    SperO = orgin.apply(lambda x : x/df[method], axis = 0)
    return SperO

pre_model_data = pd.read_csv('C:/DATA/KB_capstone/pre_model data_X.csv', engine = 'python', encoding = 'utf-8')
model_data = DataXPro_for_model(pre_model_data)

# model_data의 변수들 중 OO업종에서 유의미한 변수들만 사용!

Vfm = []

for i in Vf:
    import_features = i.columns
    df = model_data.loc[:,import_features]
    df = df.iloc[:,:-1]
    tmp = i.loc[:,['STDR_YM_CD','TRDAR_CD','SELNG_PRER_STOR']]
    df = df.merge(tmp, how = 'left', on = ['STDR_YM_CD','TRDAR_CD'])
    Vfm.append(df)  # 재현성

# model_data save

# data 저장할 때 파일명을 위해 업종 이름을 indilist에 저장
indilist = []
for index, name in enumerate(names):
    name = name.replace("_filterd.csv", "")
    indilist.append(name)

# 업종별로 데이터를 저장함
for i,df in enumerate(Vfm):
    df.to_csv('C:/DATA/KB_capstone/model_data/'+indilist[i]+'_model.csv', encoding = "ms949", index =False)
    

## 3.2 해당 상권에 영업중인 가게가 없어서 비교가 불가능한 상권: 결측상권으로 정의.
## 결측 상권과 유사한 특징을 갖는 상권을 유사상권으로 정의하고 해당 상권의 통계치를 대신하여 제공.
## 커피음료점을 대상으로 예시.

df = Vfm[-4]
df = df[df['STDR_YM_CD'] == 201804]

# 결측상권인 곳을 nan_area로 저장
nan_area = df[isnan(df['SELNG_PRER_STOR'])]

from scipy.spatial.distance import pdist, squareform

dist = pdist(df[list(df)[2:-1]].dropna(0), 'euclidean')
df_dist = pd.DataFrame(squareform(dist))

target_nan_area = 1088
target_dist = df_dist[target_nan_area-1].sort_values() # 재현성 도중 여기서 에러남
target_dist = target_dist.to_frame()
target_dist['TRDAR_CD'] = target_dist.index.values + 1 
target_dist = target_dist.merge(df[['TRDAR_CD','SELNG_PRER_STOR']], how = 'left', on = ['TRDAR_CD'])
target_dist = target_dist.dropna()
target_dist.reset_index(drop = True, inplace = True)
median = target_dist.iloc[0:11,:]['SELNG_PRER_STOR'].median()

similar_area = target_dist.iloc[0:10,:][target_dist.iloc[0:10,:]['SELNG_PRER_STOR'] == median]

## 3.3 매출 예측

train = df[~isnan(df['SELNG_PRER_STOR'])] 
test = df[isnan(df['SELNG_PRER_STOR'])]

