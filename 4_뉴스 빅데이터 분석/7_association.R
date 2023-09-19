#현재의 디렉토리주소 알아보기
getwd()
#디렌토리 주소 재설정
setwd("")
#디렌토리 주소 재설정
getwd()

#폴더 안의 파일들 확인하기
list.files()

install.packages("readxl")
library(readxl)

# 저장된 엑셀데이터 불러오기 
data <- read_excel("sample.xlsx")


install.packages("dplyr")
library(dplyr)

# 마지막 2개 변수 제외 모두 
tempdata <- data %>% select(1:17)

# 변수명을 바꾸기 
tempdata <- tempdata %>% 
  rename(
    newid = `뉴스 식별자`,
    day=일자,
    comp=언론사,
    author=기고자,
    title=제목,
    cate_all_1=`통합 분류1`,
    cate_all_2=`통합 분류2`,
    cate_all_3=`통합 분류3`,
    cate_event_1=`사건/사고 분류1`,
    cate_event_2=`사건/사고 분류2`,
    cate_event_3=`사건/사고 분류3`, 
    agent=인물,
    location=위치, 
    organ=기관,
    keyword=키워드,
    feature=특성추출,
    contents=본문
  )



install.packages("KoNLP")
library(KoNLP)

## 사전 선택 택1
useSejongDic()
useNIADic()

##tempdata에서 콘텐츠(본문) 부분만 따오기
contents <- tempdata$contents
head(contents)


txt <- extractNoun (contents)
head(txt)

#한글자는 제외합시다
txt_data <- Filter(function(x){nchar(x)>1}, txt)

##불용어처리
txt_data <- gsub("//d+","",txt_data)
txt_data <- gsub("[[:cntrl:]]","",txt_data)
txt_data <- gsub("[[:punct:]]","",txt_data)
#이녀석이 숫자를 삭제해줍니다. 
txt_data <- gsub("[[:digit:]]","",txt_data)
txt_data <- gsub("[[:lower:]]","",txt_data)
txt_data <- gsub("[[:upper:]]","",txt_data)
head(txt_data)

install.packages("tm")
library(tm)

docs <- Corpus(VectorSource(txt_data))
WordList <- sapply(docs, extractNoun, USE.NAMES=FALSE)
vectordata <- unlist(WordList)

#한글자는 제외합시다
vectordata <- Filter(function(x){nchar(x)>1}, vectordata)

aword <- unique(WordList) # 중복제거1(전체 대상)

lword <- sapply(aword, unique) # 중복제거2(줄 단위 대상) 
lword[10:15] # 추출 단어 확인

# 1) 길이가 2~4 사이의 단어 필터링 함수 정의
filter1 <- function(x){
  nchar(x) <= 8 && nchar(x) >= 2 && is.hangul(x)
}

# 2) Filter(f,x) -> filter1() 함수를 적용하여 x 벡터 단위 필터링 
filter2 <- function(x){
  Filter(filter1, x)
}

# 3) 줄 단어 대상 필터링
lword <- sapply(lword, filter2)

lword # 추출 단어 확인(길이 1개 단어 삭제됨)

# 5. 연관성 분석을 위한 트랜잭션 생성
#빈도 분석에서 tm를 사용하기 위해 Corpus를 생성해주었던 것 처럼,연관성 분석에서는 apriori를 사용하기 위해 transaction객체를 생성해주어야 한다. 
#apriori를 사용하기 위해서는 arules패키지를 설치해주어야 한다.

# arules 패키지 설치
install.packages("arules")
library(arules) 

wordtran <- as(lword, "transactions") # lword에 중복데이터가 있으면 error발생
summary(wordtran) 

# 트랜잭션 내용 보기 -> 각 트랜잭션의 단어 보기
inspect(wordtran)  

# 동일한 단어끼리 교차테이블 작성 
wordtable <- crossTable(wordtran) # 교차표 작성
wordtable
# 6. 단어 간 연관규칙 산출

tranrules <- apriori(wordtran, parameter=list(supp=0.2, conf=0.2)) 
tranrules # 연관규칙 생성 결과 보기

associ <- labels(tranrules, ruleSep=" ")
associ <-sapply(associ,strsplit," ",USE.NAMES=F)
associ <- associ[-c(1:19)]
associ

matrixassoci <- do.call("rbind", associ)
matrixassoci

install.packages("igraph")
library(igraph)

corel <- graph.edgelist(matrixassoci, directed=F)
corel

plot.igraph(corel, vertex.label=V(corel)$name, vertex.label.cex=1.0,vertex.label.color='blue', vertex.size=20, vertex.color='green', vertex.frame.color='red',)
