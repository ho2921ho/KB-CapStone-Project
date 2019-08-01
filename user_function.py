
from sklearn.preprocessing import scale, normalize,RobustScaler,MinMaxScaler 
robust = RobustScaler().fit_transform ; minmax = MinMaxScaler()
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from matplotlib import style
import matplotlib.cm as cm
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE ; tsne = TSNE(random_state=42)
from sklearn.decomposition import PCA ; pca = PCA(n_components = 2)
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import TSNE ; tsne = TSNE(random_state=42)

from collections import Counter as counter
from math import log
import pandas as pd
import numpy as np
import os
from os import listdir
import gc
from sklearn.cluster import AgglomerativeClustering
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 1000000
mpl.rcParams.update({'figure.max_open_warning': 0})
from scipy.stats.stats import pearsonr   
from sklearn import linear_model
lm = linear_model.LinearRegression()
import seaborn as sns
import scipy as sp

pd.set_option('display.max_columns', 500000)
pd.set_option('display.max_rows', 500000)

# str로 csv 파일을 불러오는 함수입니다.
def csv(name):
    file = pd.read_csv('data/' + name, engine = "python"); file.rename(columns = {"Unnamed: 0"  : "index"}, inplace = True) 
    return file  

# 업종별로 필터링해주는 함수입니다. 
def sepfilter (df,indilist):
    seplist = []
    for i in indilist:
        tmp = df[df.iloc[:,4]== i]
        seplist.append(tmp)
    return seplist

#    
def FindKsil(X_scaled,mink,maxk):
    X_tsne = tsne.fit_transform(X_scaled)
    cluster_range = range(mink,maxk)
    for n_clusters in cluster_range:
      fig, (ax1, ax2) = plt.subplots(1, 2) # Create a subplot with 1 row and 2 columns
      fig.set_size_inches(18, 7)
      ax1.set_xlim([-1, 1]) # The 1st subplot is the silhouette plot
      ax1.set_ylim([0, len(X_scaled) + (n_clusters + 1) * 10]) # The (n_clusters+1)*10 is for inserting blank space between silhouette # plots of individual clusters, to demarcate them clearly.
      clusterer = AgglomerativeClustering(n_clusters=n_clusters)  # Initialize the clusterer with n_clusters value and a random generator # seed of 10 for reproducibility.
      cluster_labels = clusterer.fit_predict( X_scaled )
      # clusters
      silhouette_avg = silhouette_score(X_scaled, cluster_labels) # The silhouette_score gives the average value for all the samples.  # This gives a perspective into the density and separation of the formed
      print("For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg)
    
      # Compute the silhouette scores for each sample
      sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)
    
      y_lower = 10
      for i in range(n_clusters):
          # Aggregate the silhouette scores for samples belonging to
          # cluster i, and sort them
          ith_cluster_silhouette_values = \
              sample_silhouette_values[cluster_labels == i]
    
          ith_cluster_silhouette_values.sort()
    
          size_cluster_i = ith_cluster_silhouette_values.shape[0]
          y_upper = y_lower + size_cluster_i
    
          color = cm.nipy_spectral(float(i)/n_clusters)
          ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)
    
          # Label the silhouette plots with their cluster numbers at the middle
          ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
          # Compute the new y_lower for next plot
          y_lower = y_upper + 10  # 10 for the 0 samples
    
      ax1.set_title("The silhouette plot for the various clusters.")
      ax1.set_xlabel("The silhouette coefficient values")
      ax1.set_ylabel("Cluster label")
    
      # The vertical line for average silhoutte score of all the values
      ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
      ax1.set_yticks([])  # Clear the yaxis labels / ticks
      ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
      # 2nd Plot showing the actual clusters formed
      colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
      ax2.scatter(X_tsne[:,0],X_tsne[:,1], marker='.', s=30, lw=0, alpha=0.7,
                  c=colors)
    
      ax2.set_title("The visualization of the clustered data.")
      ax2.set_xlabel("Feature space for the 1st coordinate")
      ax2.set_ylabel("Feature space for the 2nd coordinate")
    
      plt.suptitle(("Silhouette analysis for Agglomerative clustering on sample data "
                    "with n_clusters = %d" % n_clusters),
                   fontsize=14, fontweight='bold')
    
      plt.show() 

### 범주형 자료의 분산(?) 추정 함수
def diversity(df):
    DIVERSITY = []
    for i in range(len(df)):    
        c = [] 
        for j in range(len(list(df))):
            temp = np.repeat(j,df.iloc[i][j])
            c.extend(temp)    
        DIVERSITY.append(np.asarray(c).var())
    return DIVERSITY


# make y about sale Ysale function
def Ysale(df): 
    Yisale = df.loc[:,['STDR_YM_CD','TRDAR_CD','SVC_INDUTY_CD_NM']]
    Yisale['TRDAR_STR_SELNG_AMT'] = df['THSMON_SELNG_AMT'] / df["STOR_CO"]
    tmp = Yisale.groupby(['STDR_YM_CD','SVC_INDUTY_CD_NM']).mean()
    tmp2 = Yisale.groupby(['STDR_YM_CD','SVC_INDUTY_CD_NM']).std()
    tmp.columns = ['TRDAR_CD_f',"THSMON_STR_SELNG_AMT_MEAN"]
    tmp2.columns = ['TRDAR_CD_f2',"THSMON_STR_SELNG_AMT_STD"]
    Yisale = pd.merge(Yisale, tmp, how = 'left', on = ['STDR_YM_CD','SVC_INDUTY_CD_NM'])
    Yisale = pd.merge(Yisale, tmp2, how = 'left', on = ['STDR_YM_CD','SVC_INDUTY_CD_NM'])
    del Yisale['TRDAR_CD_f2']
    del Yisale['TRDAR_CD_f']
    return Yisale

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


def corrtable(df):
    df = df.fillna(0)
    df["SELNG_PRER_STOR"] = df["THSMON_SELNG_AMT"] / df['STOR_CO']
    X = df.iloc[:,5:]
    X.drop(X.columns[list(range(X.columns.get_loc("THSMON_SELNG_AMT"), X.columns.get_loc("AGRDE_60_ABOVE_SELNG_CO")+1))],axis=1,inplace=True)
    y = df["SELNG_PRER_STOR"]
    del X["SELNG_PRER_STOR"]
    features = X.columns.tolist() #variable name list
    target = y.name 
    correlations = {}
    for f in features:
        table = X.loc[:,f].to_frame()
        table['y']= y.copy()
        key = f + ' vs ' + target
        correlations[key] = table.corr().iloc[0,1]
        data_correlations = pd.DataFrame(correlations, index=['Value']).T
        data_correlations.sort_values(ascending = False, axis = 0, na_position = 'first', by = 'Value', inplace = True)
    return data_correlations

# 총유동인구를 단위면적당 유동인구로 변경. 

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

def Model(X,y):

    ### test, train 나누기 ..> 추후 k-fold 로
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y ,test_size=0.2,random_state = 2)
    ## reg
    regr.fit(X_train, y_train)
    ## xgb
    xgb.fit(X_train,y_train)
    
    # Calculate the Root Mean Squared Error.
    print(np.sqrt(np.mean(((y_train**2).mean() - (y_test)**2) ** 2)))
    print(np.sqrt(np.mean(((regr.predict(X_test))**2 - (y_test)**2) ** 2)))
    print(np.sqrt(np.mean(((xgb.predict(X_test))**2- (y_test)**2) ** 2)))   
    
    print(np.sqrt(np.mean((y_train.mean() - y_test) ** 2)))
    print(np.sqrt(np.mean((regr.predict(X_test) - y_test) ** 2)))
    print(np.sqrt(np.mean((xgb.predict(X_test)- y_test) ** 2)))   

    print(np.sqrt(np.mean((np.sqrt(y_train).mean() -np.sqrt(y_test)) ** 2)))
    print(np.sqrt(np.mean((np.sqrt(regr.predict(X_test)) - np.sqrt(y_test)) ** 2)))
    print(np.sqrt(np.mean((np.sqrt(xgb.predict(X_test))- np.sqrt(y_test)) ** 2)))  
    

