#현재의 디렉토리주소 알아보기
getwd()
#디렌토리 주소 재설정
setwd("")
##현재의 디렉토리주소 알아보기
getwd()

#엑셀데이터 불러보기
install.packages("readxl")
library(readxl)

#데이터 불러오기 
data <- read_excel("sample.xlsx")

#select 함수를 사용하기 위해 dplyr을 설치#
install.packages("dplyr")
library(dplyr)

#기사의 정보가 있는 16개 변인만 가지고 오기
info <- data %>% select(1:16)

# 변수명을 변경
info <- info %>% 
  rename(
    id = `뉴스 식별자`,
    date = '일자',
    company = '언론사',
    author = '기고자',
    title = '제목',
    category1 = `통합 분류1`,
    category2 = `통합 분류2`,
    category3 = `통합 분류3`,
    event1 = `사건/사고 분류1`,
    event2 = `사건/사고 분류2`,
    event3 = `사건/사고 분류3`, 
    player = '인물',
    location = '위치', 
    organization = '기관',
    keyword = '키워드',
    feature = '특성추출'
  )


# 인물들이 나온다면 네트워크 분석 흥미
# 인물 => 네트워크 생각

#각각의 인물을 쪼개기
node <- info %>% 
  separate(player, c("name1","name2","name3","name4","name5","name6","name7","name8","name9","name10"),
           sep=",",remove=F,extra="merge",fill="right") %>% 
  select(starts_with("name"))

#tudygraph라는 패키지를 사용하여 데이터프레임을 그래프 형식으로 바꾸기
#install.packages("tidygraph")
library(tidygraph)

network <-as_tbl_graph(node)
plot(network)
network


#위 plot이라는 함수는 R의 기본시각과 함수이며 네트워크 시각화에서는 흔히 ggraph라는 함수를 사용
##ggraph는 이 노드와 엣지를 구성하는 기본체계로 구성
#install.packages("ggraph")
library(ggraph)
ggraph(network) + geom_node_point() + geom_edge_link()

# 노드: 행위자 # 엣지: 연결
# 행위자와 행위자 사이 연결


#이 그래프는 stress라는 레이아웃을 자동으로 선택함
#레이아웃은 이 외에도 circle, dh, drl, fr, gem, graphopt, grid, kk, lgl, mds, kk, lgl, mds, star, ramdomly 등으로 변환할 수 있음
network %>% 
  as_tbl_graph() %>% 
  ggraph(layout='graphopt') + geom_node_text(aes(label=name)) + geom_edge_link(aes(start_cap = label_rect(node1.name), end_cap = label_rect(node2.name)))


network %>% 
  as_tbl_graph() %>% 
  mutate(cor= centrality_betweenness()) %>% 
  as_tibble %>% 
  arrange(desc(cor))

# 매개중심성: 누구와 많이 연결
