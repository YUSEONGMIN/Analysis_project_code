## 분석 프로젝트 코드

## 목차

1. [IPO(기업공개) 수 예측 분석](#1-ipo-수-예측-분석)
2. [농아인을 위한 수어 번역기 개발](#2-농아인을-위한-수어-번역기-개발)
3. [분석3](#3-분석3)

## 1. IPO 수 예측 분석

모델에 대한 구체적인 내용은 
https://github.com/YUSEONGMIN/Papers-with-code/tree/main/CSAM

```python
# 필요한 패키지를 불러옵니다.
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy.stats import skew, kurtosis
```

$$ \alpha_i \circ X = \sum_{j=1}^{X} N_j, \quad N_j \sim Poi(\alpha_i) $$

> $\circ$: Poisson Thinning 연산자  
$\alpha_i$: 0과 1 사이의 실수값  
$X$: 음이 아닌 정수값

```python
# Poisson Thinning 연산자 함수를 생성합니다.
def PT(alpha,X):
    N=np.random.poisson(alpha,X)
    Y=sum(N)
    return Y
```

$$
\begin{align*}
&X_t = \alpha_1 \circ X_{t-1}^{(1)} + \alpha_2 \circ X_{t-1}^{(2)} + \cdots + \alpha_p \circ X_{t-1}^{(p)} + \varepsilon_t, \varepsilon_t \sim Poi(\lambda)\\
&X_{t-1}^{(i)} = \frac{1}{h_i}(X_{t-1} + \cdots + X_{t-h_i})
\end{align*}
$$

> $X_t \sim$ INHAR($p$) 모델  
$X_{t-1}^{(i)}$: $h_i$의 이동평균  
$h_i$: 음이 아닌 정수값 ($1\< h_1\< ...\< h_i\< ...\< h_p$)

``` python
"""
 Args:
 hps : [h_1, h_2, ..., h_p]
 params : [alpha_1, alpha_2, ..., alpha_p, lambda]
"""

def INHAR_P(hps,params,n):
    X0=list(np.random.poisson(params[-1],hps[-1])) # 초기 값 생성
    for t in range(hps[-1],n):
        order,Y=[],[]
        for i in hps:
            globals()['Xt_{}'.format(i)] = int(np.mean(X0[t-i:t])) # X_{t-1}^{(i)} 생성
            order.append(globals()['Xt_{}'.format(i)])
        for i in range(len(hps)):
            Xt=PT(params[i],order[i]) # alpha와 h_i 이동평균으로 PT 연산자 계산
            Y.append(Xt)
        lamb=np.random.poisson(params[-1]) # 파라미터 lambda를 가진 오차항 생성
        Y=sum(Y)+lamb
        X0.append(Y)
    return X0
```

파라미터 추정 방법은 **조건부 최소제곱법**(Conditional Least Squares; CLS)을 이용했습니다.

$$
\begin{equation}
\hat \theta_{CLS} =( \hat \alpha_{1,CLS}, \dots, \hat \alpha_{p,CLS}, \,\hat \lambda _{CLS})^{\top} = \mathbb{A}^{-1}\mathbb{b}
\end{equation}
$$

$$
\mathbb{b}= \left(\sum^n_{t=1}X_t X_{t-1}^{(1)}, \dots,  \sum^n_{t=1}X_t X_{t-1}^{(p)}, \sum^n_{t=1}X_t  \right)^{\top}
$$

$$
\mathbb{A}=  \left[
\begin{array}{lllll}
\sum\limits_{t=1}(X_{t-1}^{(1)})^2 & \sum\limits_{t=1}X_{t-1}^{(2)}X_{t-1}^{(1)} & \cdots & \sum\limits_{t=1}X_{t-1}^{(p)}X_{t-1}^{(1)} & \sum\limits_{t=1}X_{t-1}^{(1)} \\
\sum\limits_{t=1}X_{t-1}^{(1)}X_{t-1}^{(2)} & \sum\limits_{t=1}(X_{t-1}^{(2)})^2 & \cdots & \sum\limits_{t=1}X_{t-1}^{(p)}X_{t-1}^{(2)} & \sum\limits_{t=1}X_{t-1}^{(2)} \\
\vdots && \ddots && \vdots \\ 
\sum\limits_{t=1}X_{t-1}^{(1)}X_{t-1}^{(p)} & \sum\limits_{t=1}X_{t-1}^{(2)}X_{t-1}^{(p)} & \cdots & \sum\limits_{t=1}(X_{t-1}^{(p)})^2 & \sum\limits_{t=1}X_{t-1}^{(p)} \\
\sum\limits_{t=1}X_{t-1}^{(1)} & \sum\limits_{t=1}X_{t-1}^{(2)} & \cdots & \sum\limits_{t=1}X_{t-1}^{(p)} & n \\
\end{array}
\right]
$$

```python
def CLSE(data,hps): # CLS Estimator
    nn,p=len(data),len(hps)
    for i in range(p+1):
        globals()['b_'+str(i)] = 0 # b 초기값 생성
    A=np.zeros((p+1,p+1)) # A 초기값 행렬 생성
    for t in range(nn-hps[-1]):
        Xt=data[t+hps[-1]] # X_t
        Xtt=[] # X_{t-1}^{(i)}
        for i in hps:
            globals()['Xt_'+str(i)] = int(np.mean(data[t+hps[-1]-i : t+hps[-1]]))
            Xtt.append(globals()['Xt_'+str(i)])
        Xtt.append(1) # [X_{t-1}^{(1)},...,X_{t-1}^{(p)},1]
        for i in range(p+1):
            globals()['b_'+str(i)] = globals()['b_'+str(i)] + Xt*Xtt[i] # sum: X_t*X_{t-1}^{(1)}, ..., X_t*1
            for j in range(p+1):
                A[i,j]=A[i,j]+Xtt[i]*Xtt[j]
    b=[]
    for i in range(p+1):
        b.append(globals()['b_'+str(i)])
    A_inv=np.linalg.inv(A)
    return np.matmul(A_inv,b) # A^{-1}*b
```


INHAR($p$) 모형은 AR($h_p$) 모형의 변형으로 볼 수 있습니다.  
이 이론을 이용하여 **Yule-Walker 추정법**으로도 파라미터를 추정할 수 있습니다.

$$
E[X_t|{\cal{F_{t-1}}}] = Y_t \\
Y_t = \alpha_1 Y_{t-1}^{(1)}+\cdots+\alpha_p Y_{t-1}^{(p)}\\
Y_t = \beta_1 Y_{t-1}+\cdots+\beta_{h_p}Y_{t-h_p}
$$

$$
\hat\alpha_{i,YW}=h_i(\frac{\sum\limits_{j=h_{i-1}+1}^{h_i}\hat\beta_{j,YW}}{h_i-h_{i-1}}-\sum_{l=i+1}^{p}\frac{\hat\alpha_{l,YW}}{h_l}), h_0=0
$$

```python
def YWE(data,hps): # Yule-Walker Estimator
    his=[0,*hps]
    beta=sm.regression.linear_model.yule_walker(data,his[-1],"mle")[0] # beta 계산
    a_h,ayw=0,[] # sum: alpha_l/h_l 초기값 0 
    for i in reversed(range(1,len(his))):
        globals()['a_'+str(i)] = his[i]*(sum(beta[his[i-1]:his[i]])/(his[i]-his[i-1])-a_h)
        a_h = a_h + globals()['a_'+str(i)]/his[i]
        ayw.append(globals()['a_'+str(i)])
    ayw.reverse()
    X,Y=INHAR(data,hps)
    lamb=np.mean([Y[t]-sum(np.array(ayw)*X[t][:len(hps)]) for t in range(len(Y))])
    return np.array([*ayw,lamb])
```

### 데이터 분석

이질적 특성을 반영한 모형이므로 이질적 시장가설을 따르는 금융 데이터를 선택했습니다.
> 참고문헌  
[Muller, U. A. et al. (1993), Fractals and Intrinsic Time - a Challenge to Econometricians, OA.](https://EconPapers.repec.org/RePEc:wop:olaswp:_009)  
[Ivanov V and Lewis CM (2008). The determinants of market-wide issue cycles for initial public offerings, JCF, 14, 567–583.](https://doi.org/10.1016/j.jcorpfin.2008.09.009)  
[Gucbilmez, U. (2015). IPO waves in China and Hong Kong, IRFA, 40, 14–26.](https://doi.org/10.1016/j.irfa.2015.05.010)  

### EDA

데이터 출처: 국내 기업공시채널 [KIND(Korea investor’s network for disclosure system)](https://kind.krx.co.kr)

2000.01 ~ 2022.07 동안의 월별 국내 기업공개(Initial public offering; IPO) 수를 분석했습니다.  
IPO 데이터는 다음과 같습니다.

|년/월|1|2|3|4|5|6|7|8|9|10|11|12|
|--|--|--|--|--|--|--|--|--|--|--|--|--|
|2022|4|13|8|7|3|10|7|0|0|0|0|0|
|2021|6|11|10|6|9|6|4|8|10|11|11|8|
|2020|2|4|7|0|2|6|15|9|11|6|6|18|
|2019|2|3|7|1|8|6|10|12|5|12|14|17|
|2018|2|8|3|2|4|4|6|9|5|7|18|22|

![IPO 그래프](https://github.com/YUSEONGMIN/Papers-with-code/raw/main/CSAM/img/ipo.png)

IPO 데이터의 정상성을 확인하기 위해 기술통계량 및 ADF-검정을 수행했습니다. 

|n|mean|med|max|min|std|skew|kurto|ADF statistic|ADF p-value|
|--|--|--|--|--|--|--|--|:-:|:-:|
271|6.830|6.000|39.000|0.000|6.104|1.985|5.408|−2.914|0.044|

```python
pd.DataFrame(ipo).describe()
skew(ipo), kurtosis(ipo, fisher=True)
adfuller(ipo)
```
 
검정 결과 유의확률(p-value) 값이 0.05보다 작으므로 정상성임을 확인할 수 있습니다.

### Fitting

월별 데이터임을 고려하여 
하이퍼 파라미터 $h_p$는 12개월 (1,12)과 6개월 (1,6,12)로 선택했습니다.

```python
hps=[1,6,12]
X,Y=INHAR(ipo,hps)
model=sm.OLS(Y,X)
results=model.fit()
results.params
CLSE(ipo,hps), YWE(ipo,hps)
```

|CLS|YW| RMSE | MAPE | SMAPE | RRSE |
| --- | --- | --- | --- | --- | --- |
| **INHAR(2)** | 2.4699 | 2.9643 | 39.1116 | 16.5429 | 1.108 |
| **INAR(2)** | 2.7350 | 3.1520 | 40.7012 | 18.4818 | 1.1781 |
||
| **INHAR(3)** | 2.4897 | 2.9743 | 39.1858 | 16.7153 | 1.1117 |
| **INAR(3)** | 2.8006 | 3.1744 | 41.3265 | 18.8416 | 1.1865 |

IPO 데이터의 CLS 추정량과 Yule-Walker 추정량을 계산하고 INHAR 모형에 적합(fitting) 시켰습니다.  
아래는 CLS 추정량을 통해 적합한 모습입니다.

![Fitting](https://github.com/YUSEONGMIN/Papers-with-code/raw/main/CSAM/img/ipo_fit.png)

### Forecasting

> In-sample: 2000.01 - 2020.12 기간의 252개 데이터  
 Out-of-sample: 2021.01 - 2022.07 기간의 19개 데이터

추정된 파라미터를 통해 예측 값을 구하고 기존의 INAR 모형과 예측 성능을 비교해보았습니다.  
95% 신뢰구간 하에 예측 그래프는 다음과 같습니다.

![Forecasting](https://github.com/YUSEONGMIN/Papers-with-code/raw/main/CSAM/img/ipo_fore.png)

```python
def INHAR_FORE(data,hps,m,n,method):
    if method=="CLS":
        F,F_cl,F_cu=[],[],[]
        for i in range(n):
            X,Y=INHAR(data[-m+i:-n+i],hps)
            model=sm.OLS(Y,X)
            results=model.fit()
            clse=results.params
            clse_cl=np.array([results.conf_int()[_][0] for _ in range(len(hps)+1)])
            clse_cu=np.array([results.conf_int()[_][1] for _ in range(len(hps)+1)])
            y,y_cl,y_cu=sum(clse*X[-1]),sum(clse_cl*X[-1]),sum(clse_cu*X[-1])
            F.append(y)
            F_cl.append(y_cl)
            F_cu.append(y_cu)
        res=np.array(F)-data[-n:]
        F_cls=[F,F_cl,F_cu]
        return res,F_cls
    
    elif method=="YW":
        F_yw=[]
        for i in range(n):
            ywe=YWE(data[-m+i:-n+i],hps)
            X,Y=INHAR(data[-m+i:-n+i],hps)
            y=sum(ywe*X[-1])
            F_yw.append(y)
        res=np.array(F_yw)-data[-n:]
        return res,F_yw  
    else:
        raise NotImplementedError
        
def PM(data,res,F): # Performance Measures
    mae=sum(abs(res))/len(res)
    rmse=np.sqrt(sum(res**2)/len(res))
    mape=sum(abs(res/data[-n:]))*100/len(res)
    smape=sum(abs(res)/(data[-n:]+abs(np.array(F))))*100/len(res)
    rrse=np.sqrt(sum(res**2)/sum((np.mean(data[-n:])-data[-n:])**2))
    return mae,rmse,mape,smape,rrse
```

| | MAE | RMSE | MAPE | SMAPE | RRSE |
| --- | --- | --- | --- | --- | --- |
| **INHAR(2)** | 2.4699 | 2.9643 | 39.1116 | 16.5429 | 1.108 |
| **INAR(2)** | 2.7350 | 3.1520 | 40.7012 | 18.4818 | 1.1781 |
||
| **INHAR(3)** | 2.4897 | 2.9743 | 39.1858 | 16.7153 | 1.1117 |
| **INAR(3)** | 2.8006 | 3.1744 | 41.3265 | 18.8416 | 1.1865 |

예측 성능 지표로 MAE, RMSE, MAPE, SMAPE, RRSE를 이용했습니다.  
INHAR 모형이 INAR 모형보다 좋은 예측 성능을 보여주었습니다.  
구체적으로 얼마나 좋은지 확인하기 위해 Efficiency를 계산하였습니다.

|Efficiency_CLS|MAE|RMSE|MAPE|SMAPE|RRSE|
|:-:|--|--|--|--|--|
|**p=2**|10.73|6.33|4.06|11.72|6.33|
|**p=3**|12.49|6.73|5.46|12.72|6.73|

Efficiency를 계산한 결과, 기존 INAR 모형보다 성능을 최대 12% 향상시킬 수 있었습니다.

#### [목차로 돌아가기](#목차)

## 2. 농아인을 위한 수어 번역기 개발

### 소목차

2-1. [데이터 수집](#2-1-데이터-수집)  
2-2. [데이터 전처리/모델링](#2-2-데이터-전처리-모델링)

### 2-1. 데이터 수집

담당 역할: 농아인 협회 위치, 국립국어원 수어사전, 한국 농아인협회 공지사항 수집


```python
## 농아인 협회/센터 위치 수집

# 필요한 패키지를 불러옵니다.
import time
import json
import sqlite3
import requests
import pandas as pd
from bs4 import BeautifulSoup

url = 'https://map.naver.com/v5/api/search'

params = {
    'caller': 'pcweb',
    'query': '농아인', # 농아인 검색
    'type': 'place',
    'searchCoord': '127.0406198501587;37.51741907323963',
    'page': '1',
    'displayCount': '20',
    'isPlaceRecommendationReplace': 'true',
    'lang': 'ko'
}

params['page'] = 1
resp = requests.get(url, params=params)
dom = BeautifulSoup(resp.text, 'html.parser')
json_obj = json.loads(resp.text)

json_obj # 기관명, 분류, x좌표, y좌표, 주소가 필요

k = 1
second=[]
while True:
    params['page'] = k
    time.sleep(1)
    resp = requests.get(url, params=params)
    json_obj = json.loads(resp.text)

    for i in json_obj['result']['place']['list']:
        first=[]
        first.append(i['name'])
        first.append(i['category'])
        first.append(i['x'])
        first.append(i['y'])
        first.append(i['address'])

        second.append(first)

    k += 1
    
    if len(json_obj['result']['place']['list']) == 0:
        break

df_1 = pd.DataFrame(second) # name, category, x, y, address가 변수로
category = df_1[1].values # 1번째 변수 = category
category = category.tolist()
category # 각 리스트마다 하나씩

for i in range(len(category)):
    element = category[i]
    ctg = ""
    for j in range(len(element)):
        ctg += element[j]
        if j != (len(element) - 1):
            ctg +=","
    category[i] = ctg
category

df_1["Category"] = category

del df_1[1]

df_1.columns = ["Name", "Longitude", "Latitude", "Address", "Category"]
order = ["Name", "Category", "Longitude", "Latitude", "Address"]
df_1 = df_1[order]

conn = sqlite3.connect('naver_map.db')
cur = conn.cursor()

cur.executescript(''' 
    DROP TABLE IF EXISTS sonmin;
''')

conn.commit()
cur = conn.cursor()

df_1.to_sql('naver_map', conn)
cur.close()

# 국립국어원 수어사전 수집

# 필요한 패키지를 불러옵니다.
import re
import requests
from bs4 import BeautifulSoup
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

url = 'http://sldict.korean.go.kr/front/sign/signList.do'

params = {
    'current_pos_index': '',
    'origin_no': '0',
    'searchWay': '',
    'top_category': '',
    'category': 'SPE001', # SPE001: 법률 용어, SPE002: 교통, ...
    'detailCategory': '',
    'searchKeyword': '',
    'pageIndex': '1'
}

params['pageIndex'] = 1
params['category'] = 'SPE001'
resp = requests.get(url, params=params,verify=False)
dom = BeautifulSoup(resp.text, 'html.parser')
dom.prettify

# 단어 카테고리 추출
dom.select('#menu > div > div > ul > li.on > ul > li > a > span')[0].text

# 단어 제목 추출
title = dom.select('#list > li > div > p > span.tit > a')[0].text
re.sub('[^가-힣]','',title)

# 단어 뜻 추출
mean = dom.select('#list > li > div > p > span.info > a > cite > span')[0].text
re.sub('[\r\n\t]','',mean)

# 단어 영상 추출
s = dom.select('#signListForm > div.result_list.mt_30 > div.wrap_list > ul > li > div.list_left > div > a > img')[1].get('src')
re.findall('MOV.+',s)[0].replace('215X161.jpg', '700X466.mp4')

# s = s.replace('215X161.jpg', '700X466.mp4')
# re.findall('MOV.+',s)[0]

# 크롤링 시작
url = 'http://sldict.korean.go.kr/front/sign/signList.do'

params = {
    'current_pos_index': '',
    'origin_no': '0',
    'searchWay': '',
    'top_category': '',
    'category': 'SPE001',
    'detailCategory': '',
    'searchKeyword': '',
    'pageIndex': '1'
}

category_num = 1
pageIndex_num = 1

category = '#menu > div > div > ul > li.on > ul > li > a > span'
title = '#list > li > div > p > span.tit > a'
mean = '#list > li > div > p > span.info > a > cite > span'
video = '#signListForm > div.result_list.mt_30 > div.wrap_list > ul > li > div.list_left > div > a > img'

while True:
    try:
        params['category'] = 'SPE'+str(category_num).zfill(3)
        while True:
            params['pageIndex'] = pageIndex_num
            resp = requests.get(url, params=params,verify=False)
            dom = BeautifulSoup(resp.text, 'html.parser')            
            for i in range(len(dom.select(title))):
                s = dom.select(video)[2*i+1].get('src')
                if len(re.findall('MOV.+',s.replace('215X161.jpg', '700X466.mp4'))) == 0:
                    continue
                f = open('add_category.txt','a')  
                f.write('\n'+dom.select(category)[category_num-1].text+
                        '\t'+dom.select(video)[2*i+1].get('src').replace('215X161.jpg', '700X466.mp4')+
                        '\t'+re.sub('[^가-힣]','',dom.select(title)[i].text)+
                        '\t'+re.sub('[\r\n\t]','',dom.select(mean)[i].text)+
                        '\t'+re.findall('MOV.+',s.replace('215X161.jpg', '700X466.mp4'))[0])
            if len(dom.select(title)) == 0:
                print('페이지 끝 \n')
                break
            pageIndex_num += 1
        category_num += 1
        pageIndex_num = 1
    except:
        print('카테고리 끝')
        f.close()
        break

## 한국 농아인협회 공지사항 수집

# 필요한 패키지를 불러옵니다.
import re
import sqlite3
import requests
import pandas as pd
from bs4 import BeautifulSoup

url = 'http://www.deafkorea.com/sub_customer/notice.php'

params = {
    'b_name': 'notice',
    'code': '',
    'keyfield': '',
    'key': '',
    'page': ''
}

params['page'] = 2
resp = requests.get(url, params=params)
resp.encoding = 'utf8'
dom = BeautifulSoup(resp.content, 'html.parser')
dom

# 공지사항 날짜
dom.select('table:not(.notice) td.w_date')

# 공지사항 제목과 주소
dom.select('table:not(.notice) a')

page_num = 1
c = []

while True:
    params['page'] = page_num
    resp = requests.get(url, params=params)
    resp.encoding = 'utf8'
    dom = BeautifulSoup(resp.content, 'html.parser')
    for a, b in zip(dom.select('table:not(.notice) a'), dom.select('table:not(.notice) td.w_date')):
        print('title:',a.text,'\ndate:',b.text,'\n',url+a['href'],'\n')
        c.extend([re.sub('[\r\n]','',a.text), b.text, url+a['href']])
    page_num += 1
    if page_num == 3:
        break

# 제목, 날짜, 주소 3개의 열을 가진 데이터프레임 만들기
n = 3
df_1 = [c[i*n : (i+1)*n] for i in range((len(c) + n - 1) // n )] 
df = pd.DataFrame.from_records(df_1,columns=('title','date','url'))

conn = sqlite3.connect('sonmin.db')
cur = conn.cursor()

cur.executescript(''' 
    DROP TABLE IF EXISTS sonmin;
    CREATE TABLE sonmin(
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    title TEXT  NOT NULL,
    date  TEXT   NOT NULL,
    url TEXT NOT NULL
    );
''')

conn.commit()
cur = conn.cursor()

sql = "INSERT INTO sonmin(title, date, url) VALUES(?,?,?)"
cur.executemany(sql, df_1) # 제목, 날짜, 주소 순으로 DB에 넣기

conn.commit()
cur.close()

conn = sqlite3.connect('sonmin.db')
cur = conn.cursor()

# 최신순으로 정렬
cur.execute("SELECT * FROM sonmin ORDER BY strftime('%Y-%m-%d', date)")
cur.fetchall()

conn.commit()
cur.close()
```





### 2-2. 데이터 전처리/모델링

담당 역할: 이미지 프레임 추출


#### [목차로 돌아가기](#목차)

## 3. 분석3

ㅁㅁㅁ

#### [목차로 돌아가기](#목차)

