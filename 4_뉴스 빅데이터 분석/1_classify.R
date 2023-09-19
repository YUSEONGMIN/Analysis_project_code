#현재의 디렉토리주소 알아보기
getwd()
#디렌토리 주소 재설정
setwd("")
##현재의 디렉토리주소 알아보기
getwd()

# 벡터(숫자만)
# 리스트(숫자와 문자)
# 행렬(숫자 층)
# 데이터프레임(숫자와 문자 층)

# 빅카인즈는 대표적인 데이터프레임


#install.packages("readxl")
library(readxl)

#데이터 불러오기 
data <- read_excel("sample.xlsx")


#select 함수를 사용하기 위해 dplyr을 설치#
#install.packages("dplyr")
library(dplyr)

#기사의 정보가 있는 16개 변인만 가지고 오기
info <- data %>% select(1:16)

# 변수명을 영어로 변경하면 오류 줄어듦
# 변수명을 변경
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

#날짜변환

#install.packages("stringr")
library(stringr)

info %>% 
  mutate(
    yr = str_sub(date,start=1,end=4),
    mn = str_sub(date,start=5,end=6),
    dy = str_sub(date,start=7,end=8)
  ) %>% select(yr:dy)

# 일자를 년/월/일 분할

# 날짜 형식의 데이터로 변환 
#install.packages("lubridate")
library('lubridate')

info %>% 
  mutate(
    yr = str_sub(date,start=1,end=4),
    mn = str_sub(date,start=5,end=6),
    dy = str_sub(date,start=7,end=8),
    date = make_date(year=yr,month=mn,day=dy)
  ) %>% select(yr:dy, date)


#검색자료에 몇 개의 언론사가 포함되어 있는 지 알아보자
table(info$company)
info %>% 
  count(company) %>% 
  arrange(desc(n))

# 언론사별 내림차순

#기고자의 분류를 알아보자
info %>% 
  mutate(
    author = str_replace(author, "/", ""),
    reporter = str_sub(author,start=1,end=3)
  ) %>% 
  count(reporter) %>% 
  arrange(desc(n))

# 기고자를 빼놓고 기사 한계점


# 분류 정보를 봅시다. 
category <- info$category1
str_split(category, pattern=">")

# > 나누기

######## 분류 좀더 하기#####
#install.packages("tidyr")
library(tidyr)

info %>% 
  separate(category1, c("ct1","ct2"),
           sep=">",remove=F,extra="merge",fill="right") %>% 
  select(category1,ct1,ct2)

# 국회_정당 _ 하위분류

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

# 정형데이터 이해 편함
# 빅카인즈 체크 (빅카인즈 활용)

