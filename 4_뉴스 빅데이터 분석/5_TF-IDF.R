# 5. TF-IDF 예제

# 필요한 패키지를 불러옵니다.
library(readxl)
library(KoNLP)

data <- read_excel("tfidf.xlsx")
contents <- data$contents
head(contents)

noun <- extractNoun(contents)

# TDM에서 Term Frequency
myCorpus <- VCorpus(VectorSource(noun))

TDM_Tf <- TermDocumentMatrix(myCorpus, control=list(removePuctuation = TRUE, removeNumbers = TRUE, stopwords = TRUE, weighting = weightTf))
TDM_TfIdf <- TermDocumentMatrix(myCorpus, control=list(removePuctuation = TRUE, removeNumbers = TRUE, stopwords = TRUE, weighting = weightTfIdf))

# 역문서 빈도

# 행렬화 (있다 / 없다)
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

# TF: Term 빈도
# DF: Documents 빈도
# IDF: log(DF/전체문서 수)
# TF-IDF: 특정문서 언급 가중치

# 모든 문서에서 모두 나오는 단어는 중요하지 않다.
# 특정 문서에서 자주 나오는 단어가 중요하다.
# 특정 문서에서 자주 언급 단어에 가중치를 주는 것 = TF-IDF

# 지소미아는 모든 문서에서 언급 -> tfidf = 0
# 쓸모 없음 -> 주제 생성 X, 네트워크 X, 분류 X
# 특정 키워드
