# 뉴스 빅데이터 분석

![News](img/News.jpg)

데이터 저널리즘이라는 새로운 시각을 배우고  
정재관 교수님과 배여운 기자님으로부터  
`R`을 활용한 뉴스 빅데이터 전처리 및 시각화 기법을 배웠습니다.

## 목차

1. [카테고리 분류](#1-기사-카테고리-분류)
2. [네트워크 분석](#2-네트워크-분석)
3. [LDA](#9-LDA)
### 1. 카테고리 분류

```R
# 필요한 패키지를 불러옵니다.
library(readxl)
library(dplyr)
library(stringr)
library(lubridate)
library(tidyr)
library(tidygraph) # 데이터프레임을 그래프로
library(ggraph) # 네트워크 분석 시각화
```

```R
# 수집한 기사 데이터를 불러옵니다.
data <- read_excel("sample.xlsx")
head(data)
```

|뉴스 식별자|일자|언론사|기고자|제목|통합 분류1|...|URL|분석제외 여부|
|-|-|-|-|-|-|-|-|-|
|011..|20191121|한겨례|박민희|"미 한반도.."|국제>국제일반|...|http..||
|081..|20191121|YTN|김상우|"日, NSC 개최.."|정치>외교|...|https..||
|||||||...|||
|081..|20191121|KBS|이세연|"여야,.."|정치>국회_정당|...|http..|중복||

수집한 기사 데이터는 위와 같습니다.  
변수는 아래 16개와	본문, URL, 분석제외 여부까지 총 19개의 변수가 있습니다.  
변수명이 한글일 경우, 오류가 날 수 있어 영어로 변환했습니다.  

```R
# 기사의 정보가 있는 16개의 변인 가져오기
info <- data %>% select(1:16)
info <- info %>% 
  rename(
    id = '뉴스 식별자',
    date = '일자',
    company = '언론사',
    author = '기고자',
    title = '제목',
    category1 = '통합 분류1',
    category2 = '통합 분류2',
    category3 = '통합 분류3',
    event1 = '사건/사고 분류1',
    event2 = '사건/사고 분류2',
    event3 = '사건/사고 분류3', 
    player = '인물',
    location = '위치', 
    organization = '기관',
    keyword = '키워드',
    feature = '특성추출'
  )
```

날짜 등을 분석할 수 있도록 변환 후,  
몇 개의 언론사가 포함되어 있는지 확인했습니다.  

```R
table(info$company)

info %>% 
  count(company) %>% 
  arrange(desc(n))
```

| company | n |
| --- | --- |
| YTN | 199 |
| 세계일보 | 118 |
| 중앙일보 | 93 |
|||

총 43개의 언론사가 있었으며, 가장 많이 수집된 언론사는 YTN으로 199개가 나왔습니다.  
이후 기사의 카테고리를 확인했습니다.  

```R
# 카테고리 정보 확인
category <- info$category1
str_split(category, pattern=">")

# 카테고리 정보 나누기
info %>% 
  separate(category1, c("ct1","ct2"),
           sep=">",remove=F,extra="merge",fill="right") %>% 
  select(category1,ct1,ct2)
```

|category1|ct1|ct2|
|-|-|-|
|국제>국제일반|국제|국제일반|
|정치>외교|정치|외교|
|정치>국회_정당|정당|국회_정당|
||||

기사의 카테고리를 세분화하여 나누었고, 그 중 "국회_정당"을 한번 더 나누었습니다.

```R
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
```

### 2. 네트워크 분석

인물이 나오면 네트워크 분석을 할 수 있습니다.  
인물(player) 변수는 다음과 같습니다.  

|뉴스 식별자|...| 인물(player) |
|-|-|-|
|011..|...| 서주석,니시노,... | 
|081..|...| 스가 요시히데 | 
|021..|...|윌터 샤프, 제임스 서먼,...|

```R
# 각각의 인물을 쪼개기
node <- info %>% 
  separate(player, c("name1","name2","name3","name4","name5","name6","name7","name8","name9","name10"),
           sep=",",remove=F,extra="merge",fill="right") %>% 
  select(starts_with("name"))
```

`ggraph`는 네트워크 시각화에서 주로 쓰이는 함수입니다.  
node와 edge를 구성하고 있습니다.  
(node: 행위자 / edge: 연결 -> 행위자와 행위자 사이를 연결)

```R
network <-as_tbl_graph(node)
plot(network)
network

ggraph(network) + geom_node_point() + geom_edge_link()
# 기본 레이아웃은 stress이며, graphopt, grid 등 다양하게 있습니다.
network %>% 
  as_tbl_graph() %>% 
  ggraph(layout='graphopt') + geom_node_text(aes(label=name)) + geom_edge_link(aes(start_cap = label_rect(node1.name), end_cap = label_rect(node2.name)))
```

![네트워크 분석](img/Network.png)

매개 중심성을 계산하면 누가 누구와 많이 연결되었는지 알 수 있습니다.  

```R
network %>% 
  as_tbl_graph() %>% 
  mutate(cor= centrality_betweenness()) %>% 
  as_tibble %>% 
  arrange(desc(cor))
```

### 3. LDA

`LDA`는 문서의 집합으로부터 어떤 토픽이 존재하는지를 알아내기 위한 알고리즘입니다.

```R
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
```

```R
# tempdata에서 콘텐츠(본문) 부분 추출
contents <- tempdata$contents
head(contents)
```

수집한 기사 내용들은 다음과 같습니다.  

||
|-|
|"한겨레-부산 국제심포지엄 이틀째인..."|
|"일본 정부는 한일..."|
|"한 일 군사정보보호..."|
||

```R
mergeUserDic(data.frame(c("개깜놀","핵존맛","JMT"), c("ncn")))
```

한글을 분석할 때  
국립국어원에서 배포하는 사전 useSejongDic()과 정보화진흥원에서 배포하는 사전 useNIADic()을 사용합니다.  
하지만 사전에 추가되지 않은 신조어를 처리해야 할 때가 있습니다.  
이 때는 mergeUserDic이라는 명령어를 써서 추가합니다.  

```R
# 명사를 추출
txt <- extractNoun (contents)
head(txt)
```

||
|-|
|"한겨레","부산","국제심포지엄","이틀째",...|
|"일본","정부","한",...|
|"한","일","군사정보보호협정",...|
||

본문 내용에서 명사를 추출했으나, 불용어가 있어 이것을 처리했습니다.  

```R
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
```

```R
docs <- Corpus(VectorSource(txt_data))

dtm <- DocumentTermMatrix(docs, control = list(removeNumbers = T,
                                               wordLength=c(2,Inf)
                                               ))
dtma <- removeSparseTerms(dtm, as.numeric(0.98))
inspect(dtma)

raw.sum <- apply(dtma,1,FUN=sum)
dtma=dtma[raw.sum!=0,]
```

|DocumentTermMatrix|documents: 1629, terms: 315|
|-|-|
|Non-/sparse entries|37114/476021|
|Sparsity|93%|
|Maximal term length|12|
|Weighting|term frequency (tf)|

|Docs/Terms|것|당|대표|미국|미아|...|
|-|-|-|-|-|-|-|
|1016|0|0|0|2|0|...|
|1021|0|0|0|1|3|...|
|254|0|3|3|0|1|...|
||||||||


```R
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
```

|topic|term|beta|
|-|-|-|
|1|가능성|0.00294|
|2|가능성|0.00095|
|3|가능성|0.00032|
|4|가능성|0.00270|
||||

```R
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
```

![LDA](img/LDA.png)

```R
spread <- topterms %>%
  mutate(topic = paste0("topic", topic)) %>%
  spread(topic, beta) %>%
  filter(topic1 > .001 | topic2 > .001) %>%
  mutate(log_ratio = log2(topic2 / topic1))

spread
```

|term|topic1|topic2|topic3|topic4|log_ratio|
|-|-|-|-|-|-|
|기자| |0.0187|0.0179| | |
|당| |0.0394| |0.0145| |
|대표| |0.0410| |0.0145| |
|||||||
|한국|0.0155|0.0172|0.0206|0.0341|0.142|
