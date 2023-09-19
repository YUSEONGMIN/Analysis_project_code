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

# 심플포스 22 09 ??

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
#install.packages("KoNLP")
library(KoNLP) # 코앤엘피: Korean National Language Processing
# 한국어 자연어 처리 # 영어는 tm

## 데이터에서 명사만 추출하고자 함. 시스템 디폴트 사전을 사용해도 되나 좀더 정확한 분석을 위해 
##국립국어원에서 배포하는 사전과 
useSejongDic()
##정보화진흥원에서 배포하는 사전을 추가로 설치
useNIADic()

# 세종딕 , 니아딕 (신뢰성)

#때로는 이 사전에 추가되지 않은 신조어를 처리해야 할 때가 있습니다.
#이 때는 mergeUserDic이라는 명령어를 써서 추가합니다.
mergeUserDic(data.frame(c("개깜놀","핵존맛","JMT"), c("ncn")))


#우리가 원하지 않는 요소들을 추출해 내야 합니다
#추출함수는 크게 세 가지가 있으며 첫번째로 gsub을 사용하겠습니다.
##gsub은 문자열의 특정 부분을 지정하여 변환하는 기능을 수행. 오피스의 CTRL + H와 같음

##불용어처리
#제어문자 삭제
txt_data <- gsub("[[:cntrl:]]","",contents)
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

# 빅카인즈 특수문자 많음
#기호 삭제
txt_data <- gsub("▲","",txt_data)
txt_data <- gsub("◎","",txt_data)

#불용어 처리는 str_replace_all이라는 구문을 이용하기도 합니다.
#이 함수를 이용하기위해 stringr라는 패키지를 인스톨합니다.

#install.packages("stringr")
library(stringr)
#영문표현삭제
newcontents <- str_replace_all(txt_data, "[[:lower:]]", "")
#제어문자 삭제
newcontents <- str_replace_all(newcontents, "[[:cntrl:]]", "")
#특수기호 삭제
newcontents <- str_replace_all(newcontents, "[[:punct:]]", "")
#숫자 = 삭제
newcontents <- str_replace_all(newcontents, "[[:digit:]]", "")
#괄호삭제
newcontents <- str_replace_all(newcontents, "\\(", "")
newcontents <- str_replace_all(newcontents, "\\)", "")

#따옴표 삭제
newcontents <- str_replace_all(newcontents, "'", "")
newcontents <- str_replace_all(newcontents, "'", "")

#특수기호 삭제
newcontents <- str_replace_all(newcontents, "[^[:print:]]", "")

# 단어만 필요하기 때문에 다른 기호나 숫자를 제거
# 숫자도 필요하다면 숫자삭제 생략

# 불용어 (우리가 사용하지 않는 단어) 삭제
# 정제 과정

#명사 추출
noun <- extractNoun(newcontents)

#텍스트마이닝 패키지 설치
#install.packages('tm')
library(tm)

#텍스트를 정제하기 위한 tm_map이라는 함수도 있습니다. tm-map은 코퍼스에서 사용됩니다. 이를위해 코퍼스로 변환합니다.

myCorpus <- Corpus(VectorSource(noun))
# 말 뭉치 형태: 코퍼스

# 불용어 삭제
# 영어가 아니므로 잘 안함
myCorpus <- tm_map(myCorpus, removePunctuation)
myCorpus <- tm_map(myCorpus, removeNumbers)
myCorpus <- tm_map(myCorpus, tolower)
myCorpus <- tm_map(myCorpus, stripWhitespace)

# 각각의 장단점 3가지 방법
# 한글 비정형데이터 처리 (깔끔하게 안됨)
# 단어 확인 -> 처음 다시 -> 반복


WordList <- sapply(myCorpus, extractNoun, USE.NAMES=FALSE)
vectordata <- unlist(WordList)
#한글자는 제외합시다
vectordata <- Filter(function(x){nchar(x)>1}, vectordata)

preview<- sort(table(vectordata), decreasing=TRUE,100)
View(preview)
# 최다빈출 읽기
# 미아 -> 지소미아 # mergeUserDict -> "지소미아"
# 문, 한자 문, 대통령 등등
# 가운데 -> 명사, 부사

#빈도추출 
wordcount <- table(vectordata)
write.csv(wordcount,file="freq.csv")

#install.packages("wordcloud")
library(wordcloud)
library(RColorBrewer)

#wordcloud
wordcloud(names(wordcount),
          freq = wordcount, #빈도량
          scale = c(3,0.5), #글자크기
          rot.per = 0.25, #회전단어 빈도
          min.freq = 10, #포함되는 최소빈도
          random.order = F, 
          random.color = F,
          colors = brewer.pal(8, "Set2"))


# 말구름1 말구름2 각각 장단점

#install.packages("wordcloud2")
library(wordcloud2)
#install.packages('devtools')
#devtools::install_github("lchiffon/wordcloud2")

wordcloud2(wordcount)
wordcloud2(wordcount, shape = "star")
letterCloud(wordcount, "A")
figPath = system.file("heart1.png", package = "wordcloud2")
wordcloud2(wordcount, figPath = "heart1.png")



#색을 랜덤하게 지정하려면 radom.color를 T로, 아니면 F로 놓고 아랫줄의 색을 적용합니다. 적용 가능학 색조합은..
?brewer.pal
#위의 태그 혹은 구글에서 'brewer.pal'로 검색이 가능합니다.

#잠깐 바그래프 형식으로 그리는 테그도 소개합니다. 알아보기 쉽게 하기 위해 최다빈출어 상위 NN개로 추출하고 이를 토대로 그리게 되는데요, 상세한 부분은 다음에 소개하겠습니다.
barplot(wordcount, las = 2, names.arg = wordcount,
        col ="lightblue", main ="최다빈출어",
        ylab = "갯수")


# 데이터 텀 매트릭스
# 단어가 있는지 없는지

# 텀 데이터 매트릭스

# 7번 단어
# 8번 단어와 단어 연결 구분

# 5번 파일 이해