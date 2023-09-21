# 8. LDA

# 필요한 패키지를 불러옵니다.
library(readxl)
library(dplyr)
library(KoNLP)
library(tm) # 텍스트마이닝 패키지
library(topicmodels)
library(tidytext)
library(ggplot2)
library(tidyr)
useSejongDic() # 국립국어원에서 배포하는 사전
useNIADic() # 정보화진흥원에서 배포하는 사전

data <- read_excel("sample.xlsx")
head(data)
tempdata <- data %>% select(1:17)
tempdata <- tempdata %>% 
  rename(
    newid = '뉴스 식별자',
    day='일자',
    comp='언론사',
    author='기고자',
    title='제목',
    cate_all_1='통합 분류1',
    cate_all_2='통합 분류2',
    cate_all_3='통합 분류3',
    cate_event_1='사건/사고 분류1',
    cate_event_2='사건/사고 분류2',
    cate_event_3='사건/사고 분류3', 
    agent='인물',
    location='위치', 
    organ='기관',
    keyword='키워드',
    feature='특성추출',
    contents='본문'
  )

# temp_data에서 콘텐츠(본문) 부분만 추출
contents <- tempdata$contents
head(contents)

# 사전에 추가되지 않은 신조어를 처리해야 할 때
# 이 때는 mergeUserDic이라는 명령어를 써서 추가합니다.
mergeUserDic(data.frame(c("개깜놀","핵존맛","JMT"), c("ncn")))

# 데이터에서 명사만 추출
txt <- extractNoun (contents)
head(txt)

# 우리가 원하지 않는 요소들을 추출
# 추출 함수 gsub 사용
# gsub은 문자열의 특정 부분을 지정하여 변환하는 기능을 수행

# 불용어 처리
# 제어문자 삭제
txt_data <- gsub("[[:cntrl:]]","",txt)
# 특수기호 삭제
txt_data <- gsub("[[:punct:]]","",txt_data)
# 숫자 삭제
txt_data <- gsub("[[:digit:]]","",txt_data)
# 소문자 삭제
txt_data <- gsub("[[:lower:]]","",txt_data)
# 대문자 삭제
txt_data <- gsub("[[:upper:]]","",txt_data)
# 특수문자 삭제
txt_data <- gsub("[^[:print:]]","",txt_data)
# 기호 삭제
txt_data <- gsub("▲","",txt_data)
txt_data <- gsub("◎","",txt_data)

head(txt_data)

docs <- Corpus(VectorSource(txt_data))

dtm <- DocumentTermMatrix(docs, control = list(removeNumbers = T,
                                               wordLength=c(2,Inf)
                                               ))
dtma <- removeSparseTerms(dtm, as.numeric(0.98))
inspect(dtma)

raw.sum <- apply(dtma,1,FUN=sum)
dtma=dtma[raw.sum!=0,]

# 전체 LDA
lda.out <-LDA(dtma,control = list(seed=100),k=4)

# 문서, 토픽
dim(lda.out@gamma)
# 토픽, 단어
dim(lda.out@beta)
# 상위 10개
top.words <- terms(lda.out, 30)
top.words

terms <- tidy(lda.out, matrix = "beta")
terms

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

spread <- topterms %>%
  mutate(topic = paste0("topic", topic)) %>%
  spread(topic, beta) %>%
  filter(topic1 > .001 | topic2 > .001) %>%
  mutate(log_ratio = log2(topic2 / topic1))

spread
