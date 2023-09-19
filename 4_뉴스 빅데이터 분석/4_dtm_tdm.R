#현재의 디렉토리주소 알아보기
getwd()
#디렌토리 주소 재설정
setwd("")
#디렌토리 주소 재설정
getwd()

#폴더 안의 파일들 확인하기
list.files()

##데이터 가져오기.
data <- readLines("kim.txt", encoding = "UTF-8")
View(data)

nstall.packages("KoNLP")
library(KoNLP)

## 사전 선택 택1
useSejongDic()
useNIADic()



install.packages("stringr")
library(stringr)
#영문표현삭제
newcontents <- str_replace_all(data, "[[:lower:]]", "")
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
txt_data <- gsub("[[:digit:]]","",txt_data)
txt_data <- gsub("[[:lower:]]","",txt_data)
txt_data <- gsub("[[:upper:]]","",txt_data)
txt_data <- gsub("[A-z]","",txt_data)
txt_data <- gsub("'","",txt_data)
txt_data <- gsub("'","",txt_data)
txt_data <- gsub("‘","",txt_data)
txt_data <- gsub("’","",txt_data)


install.packages("tm")
library(tm)
myCorpus <- Corpus(VectorSource(txt_data))

TDM_Tf <- TermDocumentMatrix(myCorpus, control=list(removePuctuation = TRUE, removeNumbers = TRUE, stopwords = TRUE, weighting = weightTf))
TDM_TfIdf <- TermDocumentMatrix(myCorpus, control=list(removePuctuation = TRUE, removeNumbers = TRUE, stopwords = TRUE, weighting = weightTfIdf))
as.matrix(TDM_Tf)
as.matrix(TDM_TfIdf)

value.tf <- as.vector(as.matrix(TDM_Tf[,]))
value.tfidf <- as.vector(as.matrix(TDM_TfIdf[,]))

#단어와 문서를 추출
doc <- rep(colnames(TDM_Tf[,]), each=dim(TDM_Tf[,])[1])
word <- rep(rownames(TDM_Tf[,]), dim(TDM_Tf[,])[2])

#모두모아 데이터프레임
valuedata <- data.frame(doc, word, value.tf, value.tfidf)
colnames(valuedata) <- c('doc', 'word', 'tf', 'tfidf')
valuedata


################ 혹은 이렇게 처리가 가능 
review_dtm_tfidf <- DocumentTermMatrix(myCorpus, control = list(weighting = weightTfIdf))
review_dtm_tfidf = removeSparseTerms(review_dtm_tfidf, 0.80)
review_dtm_tfidf
########################
#<<DocumentTermMatrix (documents: 11, terms: 256)>>
#Non-/sparse entries: 393(단어가 있는 entry)/2423(단어가 없는 entry)
#Sparsity           : 86%(sparcity가 86%인 경우 삭제)
############################

#TF와 TF-IDF 상관관계
cor.test(valuedata$tf, valuedata$tfidf, method=c("pearson", "kendall", "spearman"))

valuedata2 <- subset(valuedata, tfidf>0.3)
valuedata2

