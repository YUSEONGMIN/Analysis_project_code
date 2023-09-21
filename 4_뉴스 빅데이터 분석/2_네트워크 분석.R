2. 분류_네트워크

# 필요한 패키지를 불러옵니다.
library(readxl)
library(dplyr)
library(tidygraph) # 데이터프레임을 그래프로
library(ggraph)

data <- read_excel("sample.xlsx")
info <- data %>% select(1:16)
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

# 인물이 나오면 네트워크 분석 가능

# 각각의 인물을 쪼개기
node <- info %>% 
  separate(player, c("name1","name2","name3","name4","name5","name6","name7","name8","name9","name10"),
           sep=",",remove=F,extra="merge",fill="right") %>% 
  select(starts_with("name"))

network <-as_tbl_graph(node)
plot(network)
network

# 네트워크 시각화에서 주로 쓰는 함수 (node와 edge를 구성)
# node: 행위자 / edge: 연결 -> 행위자와 행위자 사이를 연결
ggraph(network) + geom_node_point() + geom_edge_link()

# 기본 레이아웃은 stress이며, graphopt, grid 등 다양하게 있습니다.
network %>% 
  as_tbl_graph() %>% 
  ggraph(layout='graphopt') + geom_node_text(aes(label=name)) + geom_edge_link(aes(start_cap = label_rect(node1.name), end_cap = label_rect(node2.name)))

# 매개중심성: 누구와 많이 연결되었는지
network %>% 
  as_tbl_graph() %>% 
  mutate(cor= centrality_betweenness()) %>% 
  as_tibble %>% 
  arrange(desc(cor))
