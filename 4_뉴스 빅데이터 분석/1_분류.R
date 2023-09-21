# 1. 분류

# 필요한 패키지를 불러옵니다.
library(readxl)
library(dplyr)
library(stringr)
library(lubridate)
library(tidyr)

data <- read_excel("sample.xlsx")
head(data)

# 기사의 정보가 있는 16개의 변인 가져오기
info <- data %>% select(1:16)
info <- info %>% 
  rename(
    id = `뉴스 식별자`,
    date = '일자',
    company = '언론사',
    author = '기고자',
    title = '제목',
    category1 = `통합 분류1`,
    category2 = `통합 분류2`,
    category3 = `통합 분류3`,
    event1 = `사건/사고 분류1`,
    event2 = `사건/사고 분류2`,
    event3 = `사건/사고 분류3`, 
    player = '인물',
    location = '위치', 
    organization = '기관',
    keyword = '키워드',
    feature = '특성추출'
  )

# 날짜 변환
info %>% 
  mutate(
    yr = str_sub(date,start=1,end=4),
    mn = str_sub(date,start=5,end=6),
    dy = str_sub(date,start=7,end=8)
  ) %>% select(yr:dy)

# 년/월/일 분할
info %>% 
  mutate(
    yr = str_sub(date,start=1,end=4),
    mn = str_sub(date,start=5,end=6),
    dy = str_sub(date,start=7,end=8),
    date = make_date(year=yr,month=mn,day=dy)
  ) %>% select(yr:dy, date)


# 몇 개의 언론사가 포함되어 있는지 확인
table(info$company)

info %>% 
  count(company) %>% 
  arrange(desc(n))

# 어느 기자 분이 기사를 많이 썼는지 확인
info %>% 
  mutate(
    author = str_replace(author, "/", ""),
    reporter = str_sub(author,start=1,end=3)
  ) %>% 
  count(reporter) %>% 
  arrange(desc(n))

# 기자를 빼놓고 기사를 씀 (한계점)

# 분류 정보 확인
category <- info$category1
str_split(category, pattern=">")

# 분류 정보 나누기
info %>% 
  separate(category1, c("ct1","ct2"),
           sep=">",remove=F,extra="merge",fill="right") %>% 
  select(category1,ct1,ct2)

# 정치>국회_정당 하위 분류
info %>% 
  separate(category1, c("ct1","ct2"),
           sep=">",remove=F,extra="merge",fill="right") %>% 
  separate(ct1, c("ct1a","ct1b"),
           sep="_",remove=F,extra="merge",fill="right") %>% 
  separate(ct2, c("ct2a","ct2b"),
           sep="_",remove=F,extra="merge",fill="right") %>% 
  select(category1,starts_with("ct")) 

firstct <- info %>% 
  separate(category1, c("ct1","ct2"),
           sep=">",remove=F,extra="merge",fill="left") %>% 
  separate(ct1, c("ct1a","ct1b"),sep="_",remove=F,extra="merge",fill="left") %>% 
  separate(ct2, c("ct2a","ct2b"),sep="_",remove=F,extra="merge",fill="left") %>% 
  select(ct1a, ct1b, ct2a,ct2b) 

table(c(firstct$ct1a,firstct$ct1b,firstct$ct2a,firstct$ct2b))  
