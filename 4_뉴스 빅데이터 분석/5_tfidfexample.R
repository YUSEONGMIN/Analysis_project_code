# 4번과 5번은 데이터가 다르며
# 4번이 조금 더 쉬움


#현재의 디렉토리주소 알아보기
getwd()
#디렌토리 주소 재설정
setwd("C:/Users/dbcjj/Desktop/bigkinds")
#디렌토리 주소 재설정
getwd()

#install.packages("readxl")
library(readxl)

# 저장된 엑셀데이터 불러오기 
data <- read_excel("tfidf.xlsx")

#install.packages("KoNLP")
library(KoNLP)

contents <- data$contents
head(contents)

noun <- extractNoun(contents)

# tdm 에서 term frequency
myCorpus <- VCorpus(VectorSource(noun))

TDM_Tf <- TermDocumentMatrix(myCorpus, control=list(removePuctuation = TRUE, removeNumbers = TRUE, stopwords = TRUE, weighting = weightTf))
TDM_TfIdf <- TermDocumentMatrix(myCorpus, control=list(removePuctuation = TRUE, removeNumbers = TRUE, stopwords = TRUE, weighting = weightTfIdf))
# 영문서 빈도

# 행렬화 (있다 / 없다)
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


# TF (Term 빈도)
# DF(Documents 빈도)
# IDF(log(DF/전체문서 수))
# TFIDF(특정문서 언급 가중치)

# 모든문서에서 모두 나오는 단어는 중요하지 않다
# 특정문서에서 자주 나오는 단어가 중요하다
# 특정문서 자주 언급 단어에 가중치를 주는 것 TFIDF

# 지소미아 모든 문서 언급 -> tfidf = 0
# 쓸모없음 -> 주제 생성 X , 네트워크 X , 분류 X
# 특정 키워드