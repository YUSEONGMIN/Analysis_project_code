# 3. 내용 분석

# 필요한 패키지를 불러옵니다.
library(readxl)
library(dplyr)
library(KoNLP) # Korean National Language Processing
library(stringr)
library(tm)
library(wordcloud)
library(RColorBrewer)
library(wordcloud2)
# install.packages('devtools')
# devtools::install_github("lchiffon/wordcloud2")
useSejongDic()
useNIADic()

data <- read_excel("sample.xlsx")
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

contents <- tempdata$contents
head(contents)

mergeUserDic(data.frame(c("개깜놀","핵존맛","JMT"), c("ncn")))

txt_data <- gsub("[[:cntrl:]]","",contents)
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

txt_data <- gsub("▲","",txt_data)
txt_data <- gsub("◎","",txt_data)

# 영문표현 삭제
newcontents <- str_replace_all(txt_data, "[[:lower:]]", "")
# 제어문자 삭제
newcontents <- str_replace_all(newcontents, "[[:cntrl:]]", "")
# 특수기호 삭제
newcontents <- str_replace_all(newcontents, "[[:punct:]]", "")
# 숫자 = 삭제
newcontents <- str_replace_all(newcontents, "[[:digit:]]", "")
# 괄호 삭제
newcontents <- str_replace_all(newcontents, "\\(", "")
newcontents <- str_replace_all(newcontents, "\\)", "")
# 따옴표 삭제
newcontents <- str_replace_all(newcontents, "'", "")
newcontents <- str_replace_all(newcontents, "'", "")
# 특수기호 삭제
newcontents <- str_replace_all(newcontents, "[^[:print:]]", "")

noun <- extractNoun(newcontents)

myCorpus <- VCorpus(VectorSource(noun))

myCorpus <- tm_map(myCorpus, removePunctuation)
myCorpus <- tm_map(myCorpus, removeNumbers)
myCorpus <- tm_map(myCorpus, tolower)
myCorpus <- tm_map(myCorpus, stripWhitespace)

myCorpus <- Corpus(VectorSource(noun))

WordList <- sapply(myCorpus, extractNoun, USE.NAMES=FALSE)
vectordata <- unlist(WordList)

vectordata <- Filter(function(x){nchar(x)>1}, vectordata)

preview<- sort(table(vectordata), decreasing=TRUE,100)
View(preview)

wordcount <- table(vectordata)
write.csv(wordcount,file="freq.csv")

wordcloud(names(wordcount),
          freq = wordcount, # 빈도량
          scale = c(3,0.5), # 글자 크기
          rot.per = 0.25, # 회전단어 빈도
          min.freq = 10, # 포함되는 최소빈도
          random.order = F, 
          random.color = F,
          colors = brewer.pal(8, "Set2"))

wordcloud2(wordcount)
wordcloud2(wordcount, shape = "star")
letterCloud(wordcount, "A")
figPath = system.file("heart1.png", package = "wordcloud2")
wordcloud2(wordcount, figPath = "heart1.png")

?brewer.pal

barplot(wordcount, las = 2, names.arg = wordcount,
        col ="lightblue", main ="최다빈출어",
        ylab = "갯수")
