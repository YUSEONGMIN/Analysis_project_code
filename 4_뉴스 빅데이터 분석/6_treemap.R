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


nstall.packages("KoNLP")
library(KoNLP)

## 사전 선택 택1
useSejongDic()
useNIADic()

install.packages("tm")
library(tm)

##tempdata에서 콘텐츠(본문) 부분만 따오기
contents <- data$'본문' 
head(contents)


install.packages("stringr")
library(stringr)
#영문표현삭제
newcontents <- str_replace_all(contents, "[[:lower:]]", "")
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

noun <- extractNoun(newcontents)

##불용어처리
txt_data <- gsub("//d+","",noun)
txt_data <- gsub("[[:cntrl:]]","",txt_data)
txt_data <- gsub("[[:punct:]]","",txt_data)
#이녀석이 숫자를 삭제해줍니다. 
txt_data <- gsub("[[:digit:]]","",txt_data)
txt_data <- gsub("[[:lower:]]","",txt_data)
txt_data <- gsub("[[:upper:]]","",txt_data)
txt_data <- gsub("[A-z]","",txt_data)
txt_data <- gsub("'","",txt_data)
txt_data <- gsub("'","",txt_data)
txt_data <- gsub("‘","",txt_data)
txt_data <- gsub("’","",txt_data)
head(txt_data)

myCorpus <- Corpus(VectorSource(txt_data))

myCorpus <- tm_map(myCorpus, removePunctuation)
myCorpus <- tm_map(myCorpus, removeNumbers)
myCorpus <- tm_map(myCorpus, tolower)
myCorpus <- tm_map(myCorpus, stripWhitespace)

WordList <- sapply(myCorpus, extractNoun, USE.NAMES=FALSE)
vectordata <- unlist(WordList)
vectordata <- Filter(function(x){nchar(x)>1}, vectordata)

preview<- sort(table(vectordata), decreasing=TRUE,100)
View(preview)

#빈도추출 
wordcount <- table(vectordata)
write.csv(wordcount,file="freq.csv")


#상위빈도로 정렬해서 result2로 명명
result2 <- sort(wordcount, decreasing=TRUE)
View(result2)

#누적빈도 알아보기
cumsum.word.freq <-cumsum(result2)
cumsum.word.freq[1:50]

#전체합이1이되는비율알아보기
prop.word.freq <-cumsum.word.freq/cumsum.word.freq[length(cumsum.word.freq)]
prop.word.freq[1:100]

#상위빈도 단어는 따로 저장
result1 <- head(sort(wordcount, decreasing=TRUE), 100)


#데이터프레임으로 변환
result1_wf <-data.frame(result1)
View(result1_wf)
library('RColorBrewer')
display.brewer.all()

pal<-brewer.pal(6,"Dark2")

#트리맵그리기
install.packages("treemap")
library(treemap)
dev.off()
treemap(result1_wf # 대상 데이터 설정
        ,title = "지소미아"
        ,index = "vectordata" # 박스 안에 들어갈 변수 설정
        ,vSize = "Freq"  # 박스 크기 기준
        ,fontfamily.labels = "Gothic" # 맥 폰트 설정
        ,fontsize.labels = 10 # 폰트 크기 설정
        ,palette=pal # 위에서 만든 팔레트 정보 입력
        ,border.col = "black")
