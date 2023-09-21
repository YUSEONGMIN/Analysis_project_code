# 4. DTM_TDM

# 필요한 패키지를 불러옵니다.
library(KoNLP)
library(stringr)
library(tm)
useSejongDic()
useNIADic()

data <- readLines("kim.txt", encoding = "UTF-8")
View(data)

# 영문표현 삭제
newcontents <- str_replace_all(data, "[[:lower:]]", "")
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

noun <- extractNoun(newcontents)

# 불용어 처리
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

myCorpus <- Corpus(VectorSource(txt_data))

TDM_Tf <- TermDocumentMatrix(myCorpus, control=list(removePuctuation = TRUE, removeNumbers = TRUE, stopwords = TRUE, weighting = weightTf))
TDM_TfIdf <- TermDocumentMatrix(myCorpus, control=list(removePuctuation = TRUE, removeNumbers = TRUE, stopwords = TRUE, weighting = weightTfIdf))
as.matrix(TDM_Tf)
as.matrix(TDM_TfIdf)

value.tf <- as.vector(as.matrix(TDM_Tf[,]))
value.tfidf <- as.vector(as.matrix(TDM_TfIdf[,]))

# 단어와 문서를 추출
doc <- rep(colnames(TDM_Tf[,]), each=dim(TDM_Tf[,])[1])
word <- rep(rownames(TDM_Tf[,]), dim(TDM_Tf[,])[2])

# 데이터프레임
valuedata <- data.frame(doc, word, value.tf, value.tfidf)
colnames(valuedata) <- c('doc', 'word', 'tf', 'tfidf')
valuedata

# 다른 방식으로 처리
review_dtm_tfidf <- DocumentTermMatrix(myCorpus, control = list(weighting = weightTfIdf))
review_dtm_tfidf = removeSparseTerms(review_dtm_tfidf, 0.86)
review_dtm_tfidf

# TF와 TF-IDF의 상관관계
cor.test(valuedata$tf, valuedata$tfidf, method=c("pearson", "kendall", "spearman"))

valuedata2 <- subset(valuedata, tfidf>0.3)
valuedata2
