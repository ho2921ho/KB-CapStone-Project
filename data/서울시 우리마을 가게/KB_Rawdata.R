setwd("C:/Users/renz/Documents/GitHub/KB-CapStone/data/서울시 우리마을 가게")
library(dplyr)

### 소득소비 2017 ~ 2018.05
#### 상권
in_cnsmp <- read.csv("서울시 우리마을가게 상권분석서비스(상권-소득소비).csv", sep= "\t")
in_cnsmp <- in_cnsmp %>% filter(in_cnsmp$기준_년월_코드 >= 201701)
in_cnsmp <- in_cnsmp %>% filter(in_cnsmp$상권_코드 <= 1744)

#### 배후지 
out_cnsmp <- read.csv("서울시 우리마을가게 상권분석서비스(상권배후지-소득소비).csv",sep= "\t")
out_cnsmp <- out_cnsmp %>% filter(out_cnsmp$기준년월코드>= 201701)
out_cnsmp <- out_cnsmp %>% filter(out_cnsmp$상권코드 <= 1744)

#### 상권 수 확인 
length(unique(in_cnsmp$상권_코드))
length(unique(out_cnsmp$상권코드))

### 아파트 2017 ~ 2018.05
in_apt<- read.csv("서울시 우리마을가게 상권분석서비스(상권-아파트).csv")
in_apt <- in_apt %>% filter(in_apt$기준_년월_코드 >= 201701)
in_apt <- in_apt %>% filter(in_apt$상권_코드 <= 1744)

out_apt <- read.csv("서울시 우리마을가게 상권분석서비스(상권배후지-아파트).csv")
out_apt <- out_apt %>% filter(out_apt$기준년월.코드 >= 201701)
out_apt <- out_apt %>% filter(out_apt$상권.코드 <= 1744)

#### 상권 수 확인 
length(unique(in_apt$상권_코드))
length(unique(out_apt$상권.코드))

#### 결측 상권코드. 
unique = unique(out_apt$상권.코드)
index = c(1:1744)
find = rep(0,1744)
for (i in index) {
  find[i] = index[i] %in% unique
}
which(find == 0)

### 상권 지수지표.

in_index <- read.csv("서울시 우리마을가게 상권분석서비스(상권-지수지표).csv")
in_index <- in_index %>% filter(in_index$기준_년월_코드>= 201701)
in_index <- in_index %>% filter(in_index$상권_코드 <= 1744)

#### 상권 수 확인
length(unique(in_index$상권_코드))

#### 결측된 상권코드. 
unique = unique(in_index$상권_코드)
index = c(1:1744)
find = rep(0,1744)
for (i in index) {
  find[i] = index[i] %in% unique
}
which(find == 0)

table(in_index$서비스_업종_코드_명)
