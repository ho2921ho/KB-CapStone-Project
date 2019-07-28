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
    Vfm.append(df)

# model_data save

indilist = []
for index, name in enumerate(names):
    name = name.replace("_filterd.csv", "")
    indilist.append(name)
    
for i,df in enumerate(Vfm):
    df.to_csv('C:/DATA/KB_capstone/model_data/'+indilist[i]+'_model.csv', encoding = "ms949", index =False)
    

## 3.2 해당 상권에 영업중인 가게가 없어서 비교가 불가능한 상권: 결측상권으로 정의.
## 결측 상권과 유사한 특징을 갖는 상권을 유사상권으로 정의하고 해당 상권의 통계치를 대신하여 제공.
## 커피음료점을 대상으로 예시.

df = Vfm[-4]
df = df[df['STDR_YM_CD'] == 201804]

nan_area = df[isnan(df['SELNG_PRER_STOR'])]

from scipy.spatial.distance import pdist, squareform

dist = pdist(df[list(df)[2:-1]].dropna(0), 'euclidean')
df_dist = pd.DataFrame(squareform(dist))

target_nan_area = 1088
target_dist = df_dist[target_nan_area-1].sort_values()
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

