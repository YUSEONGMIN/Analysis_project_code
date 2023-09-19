#현재의 디렉토리주소 알아보기
getwd()
#디렌토리 주소 재설정
setwd("")
#디렌토리 주소 재설정
getwd()

#폴더 안의 파일들 확인하기
list.files()

#엑셀파일을 읽을 수 있는 패키지 다운 및 실행
install.packages("readxl")
library(readxl)

# 저장된 엑셀데이터 불러오기 
data <- read_excel("sample.xlsx")
head(data)

#파이프 함수를 이용하여 변인 선택을 위한 dplyr 패키지 실행
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

##tempdata에서 콘텐츠(본문) 부분만 따오기
contents <- tempdata$contents
head(contents)


#만약 데이터파일이 텍스트라면...다음 예제에서 해보죠

#한국어 텍스트분석을 위한 패키지
install.packages("KoNLP")
library(KoNLP)

## 데이터에서 명사만 추출하고자 함. 시스템 디폴트 사전을 사용해도 되나 좀더 정확한 분석을 위해 
##국립국어원에서 배포하는 사전과 
useSejongDic()
##정보화진흥원에서 배포하는 사전을 추가로 설치
useNIADic()

#때로는 이 사전에 추가되지 않은 신조어를 처리해야 할 때가 있습니다.
#이 때는 mergeUserDic이라는 명령어를 써서 추가합니다.
mergeUserDic(data.frame(c("개깜놀","핵존맛","JMT"), c("ncn")))

#그럼 일단 명사를 추출해 봅시다
txt <- extractNoun (contents)
head(txt)

#우리가 원하지 않는 요소들을 추출해 내야 합니다
#추출함수는 크게 세 가지가 있으며 첫번째로 gsub을 사용하겠습니다.
##gsub은 문자열의 특정 부분을 지정하여 변환하는 기능을 수행. 오피스의 CTRL + H와 같음

##불용어처리
#제어문자 삭제
txt_data <- gsub("[[:cntrl:]]","",txt)
#특수기호 삭제
txt_data <- gsub("[[:punct:]]","",txt_data)
#숫자 삭제
txt_data <- gsub("[[:digit:]]","",txt_data)
#소문자 삭제
txt_data <- gsub("[[:lower:]]","",txt_data)
#대문자 삭제
txt_data <- gsub("[[:upper:]]","",txt_data)
#특수문자 삭제
txt_data <- gsub("[^[:print:]]","",txt_data)

#기호 삭제
txt_data <- gsub("▲","",txt_data)
txt_data <- gsub("◎","",txt_data)

#나중에 실행하세요
head(txt_data)

#텍스트마이닝 패키지 설치
install.packages("tm")
library(tm)

docs <- Corpus(VectorSource(txt_data))







dtm <- DocumentTermMatrix(docs, control = list(removeNumbers = T,
                                               wordLength=c(2,Inf)
                                               ))
dtma <- removeSparseTerms(dtm, as.numeric(0.98))
inspect(dtma)

######################################3
raw.sum <- apply(dtma,1,FUN=sum)
dtma=dtma[raw.sum!=0,]


install.packages('topicmodels')
library(topicmodels)
##전체 LDA##
lda.out <-LDA(dtma,control = list(seed=100),k=4)

#문서, 토픽
dim(lda.out@gamma)
#토픽, 단어
dim(lda.out@beta)
#상위 10개
top.words <- terms(lda.out, 30)
top.words

######################################
library(tidytext)

terms <- tidy(lda.out, matrix = "beta")
terms

library(ggplot2)
library(dplyr)

topterms <- terms %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

topterms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()


######################
library(tidyr)

spread <- topterms %>%
  mutate(topic = paste0("topic", topic)) %>%
  spread(topic, beta) %>%
  filter(topic1 > .001 | topic2 > .001) %>%
  mutate(log_ratio = log2(topic2 / topic1))

spread
