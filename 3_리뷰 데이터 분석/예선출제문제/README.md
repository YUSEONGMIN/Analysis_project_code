# 텍스트 감성 분석 문제

## 목차
1. [문제 정의](#1-문제-정의)
2. [탐색적 자료분석](#2-탐색적-자료분석)
3. [모델링](#3-모델링)
4. [정리](#4-정리)

## 1. 문제 정의

![problem](img/2_camp_exam.jpg)

리뷰 글을 분석하여 긍정/부정을 분류하는 문제입니다.  


## 2. 탐색적 자료분석

### 탐색적 자료분석

데이터의 특성을 알아보기 위해 탐색적 자료분석(EDA)를 했습니다.  

```python
필요한 패키지를 불러옵니다.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
total = train.append(test, ignore_index = True)
total.shape
```

|id|정말|너무|...|그럼|label|
|-|-|-|-|-|-|
|1|0|0|...|0|1|
||||||
|4|1|0|...|0|0|
||||||

> (18452, 302)

문제에 정의된대로 Train 데이터는 9452개, Test 데이터는 9000개이며,  
300개의 Word와 ID, label 변수를 포함해 총 302개의 열이 존재합니다.

```python
X = total.drop(columns = ['Id', 'label'])
np.unique(X)
```
> array([0, 1], dtype=int64)

ID와 label을 제외한 모든 Feature는 0과 1로 이루어진 `Binary` 데이터 입니다.  
각 행은 문서를 의미하며, 이 행렬을 `Binary Term Dcoumnet` 행렬이라고 합니다.  

문서가 몇 개의 Word를 가지고 있는지 파악하기 위해 박스플롯을 그렸습니다.  

```python
# Document의 Token 갯수의 Distibution 확인
sns.boxplot(y = X.sum(axis = 1))
```

![boxplot](img/boxplot.png)


```python
# Document의 Token 갯수의 요약통계량 확인
X.sum(axis = 1).describe()
```




```python
# 가장 많이 등장한 Token 30개 확인
pd.DataFrame(X.sum(axis = 0)).sort_values(0, ascending=False).iloc[:30]
```


## 3. 모델링

분류 모델로 `로지스틱 회귀모델`과 `K-최근접 이웃(KNN) 모델`을 이용했습니다.  

```python
# 필요한 패키지를 불러옵니다.
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

```python
algorithmes = [LogisticRegression(), KNeighborsClassifier(n_jobs=-1)]

# 실험 파라미터 설정
params = []

# Logistic Regression 하이퍼 파라미터
params.append([{
    "solver" : ["saga"],
    "penalty" : ["l1"],
    "C" : [0.1,  5.0, 7.0, 10.0, 15.0, 20.0, 100.0]
    },{
    "solver" : ['liblinear'],
    "penalty" : ["l2"],
    "C" : [0.1,  5.0, 7.0, 10.0, 15.0, 20.0, 100.0]
    }
    ])

# KNN 하이퍼 파라미터
params.append({
    "p":[int(i) for i in range(1,3)],
    "n_neighbors":[i for i in range(2, 6)]})

# 5 - Fold Cross Validation & Accuracy
scoring = ['accuracy']
estimator_results = []
for i, (estimator, params) in enumerate(zip(algorithmes,params)):
    gs_estimator = GridSearchCV(
            refit="accuracy", estimator=estimator,param_grid=params, scoring=scoring, cv=5, verbose=1, n_jobs=4)
    print(gs_estimator)

    gs_estimator.fit(X, y)
    estimator_results.append(gs_estimator)
```

Grid Search 방법으로 모델의 하이퍼 파라미터를 찾고,  
5-Fold 교차 검증을 통해 가장 좋았던 파라미터를 찾았습니다.  

로지스틱 회귀모델의 가장 좋은 성능은 **0.7896** 이며, KNN 모델의 가장 좋은 성능은 **0.6977** 로 나왔습니다.  
가장 성능이 좋은 로지스틱 회귀모델로 학습을 진행했습니다.  

```python
# Logistic를 통한 가장 좋은 Feature 확인
feature_name = X.columns.to_numpy()

print("Coefficient가 가장 큰 30개의 Feature 확인")
feature_name[estimator_results[0].best_estimator_.coef_.argsort()[::-1]][:30]
```

```python
#가장 좋은 모델 설정
model = estimator_results[0].best_estimator_
model.fit(X_train, y_train)
```


```python
# validation 예측
pred = model.predict(X_vld)

# validation set 성능 확인
print(classification_report(y_vld, pred))
print(confusion_matrix(y_vld, pred))
```


```python
# 전체데이터로 학습하기
model.fit(X, y)

# 테스트 데이터 복사
X_test = test.copy()
final_pred = model.predict(X_test)
```




## 4. 정리

