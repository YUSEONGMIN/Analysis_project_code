# install.packages(c("tidyverse","rvest",'lubridate',"glue","N2H4"),dependencies=T)

library(tidyverse)
library(rvest)
library(lubridate)
library(glue)
library(N2H4)
       
# tidy는 dply같은 패키지 %>%  가능          
# rvest는 웹
# lubridate 시계열
# glue 문자열 붙일 때
# N2H4 네이버 기사 가져올 때
# dependency 연관된 패키지까지 설치

url <- "http://www.korea2me.com/country/KR"
# 대한민국과 다른 국가간의 거리
# div 속성을 알려줌
# a 태그
# href 속성에 주소 붙음

# rvest
h <- read_html(url) 

# href 속성 값 가져오기
# /cc/KR-국가

# 특정 노드 가져오기
href <- h %>% 
  html_nodes("div.col-md-6 > a") %>%
  html_attr("href")

h %>% html_nodes("a")
# 불필요한 모든 a 태그 가져옴 (조건 필요 > 사용 하위)
#copy selector
#tocity > div > div:nth-child(1) > a:nth-child(1)

# # ID # . class
# div class = -> div.
# div id = -> div# 

# nodes 태그
# <a href 속성>

# class 속성명
# html_attr("class")
# id 속성-> "id" 

href<-str_remove_all(href,"/cc/")
base<-"http://www.korea2me.com/cc/"

base_url<-str_c(base,href)
base_url
# 같은 기능 paste



test<-read_html(base_url[1])
value<-test %>% 
  html_nodes("h1.h3") %>% 
  html_text()

# h1 태그 class 명 h3
# <h1 class 속성 > 텍스트 <>

# 표 태그
# copy selector

value2<- test %>% 
  html_node("body > div:nth-child(2) > div > div.col-md-8 > table.table.table-condensed2.em12.resp-table > tbody > tr:nth-child(1) > td:nth-child(1)") %>% 
  html_text()

df_final<-NULL
for (i in base_url){
  print(i)
  base_h<-read_html(i)
  
  base_text<-base_h %>% 
    html_nodes("h1.h3") %>% 
    html_text()
  
  base_value<-base_h %>% 
    html_node("body > div:nth-child(2) > div > div.col-md-8 > table.table.table-condensed2.em12.resp-table > tbody > tr:nth-child(1) > td:nth-child(1)") %>% 
    html_text()
  
  df_local<-data.frame(base_text,base_value)
  df_final<-rbind(df_final,df_local)
}

write.csv(df_final,"df_final.csv")