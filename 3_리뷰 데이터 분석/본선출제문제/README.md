# 이미지 다중 분류 문제

## 목차
1. [문제 정의](#1-문제-정의)
2. [머신러닝](#2-머신러닝)
3. [딥러닝](#3-딥러닝)
4. [정리](#4-정리)

## 1. 문제 정의

![problem](img/1_camp_exam.png)

리뷰 사진을 이용하여 만두, 새우튀김, 순대를 분류하는 문제입니다.  

머신러닝 데이터셋은 `pickle` 파일로 제공되었고,  
딥러닝 데이터셋은 `numpy` 파일로 제공되었습니다.  

## 2. 머신러닝

### 탐색적 자료분석

데이터의 특성을 알아보기 위해 탐색적 자료분석(EDA)를 했습니다.  

```python
필요한 패키지를 불러옵니다.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

```python
train.shape, test.shape
train.head(5)
# 3072개의 픽셀과 label
```
> ((600,3073), (300, 3072))

| |0|1|...|3071|label|
|-|-|-|-|-|-|
|0|0.160784|0.121569|...|0.003922|3|
|1|0.964706|0.964706|...|0.952941|3|
|2|	0.564706|0.415686|...|0.411765|3|
|||||||

각 행은 각각의 이미지를 나타내며,    
총 600개의 이미지로 3072개의 픽셀과 label이 있습니다.  
3072개의 픽셀을 (32,32,3) 형태로 변환하면 이미지가 됩니다.  

이미지로 변환 후, 각 label의 음식을 확인해보았습니다.  

| 만두 | 새우튀김 | 순대 |
| --- | --- | --- |
| ![Mandoo](img/Mandoo.png) | ![Shrimp](img/Shrimp.png) | ![Sundae](img/Sundae.png) |
| label 3 | label 4 | label 5 |
| 1~200번째 | 201~400번째 | 401~600번째 |

```python
from sklearn.model_selection import train_test_split
X_train, X_vld, y_train, y_vld = train_test_split(X, y, random_state=42, test_size = .2)
X_train.shape, X_vld.shape, y_train.shape, y_vld.shape
```
> ((480, 3072), (120, 3072), (480,), (120,))

600개의 이미지 중 480장으로 학습에 이용하였고,  
나머지 120장은 Validation으로 구성하여 성능을 검증하는데 이용했습니다.

### 모델링

분류 모델로 `로지스틱 회귀모델`과 `K-최근접 이웃(KNN) 모델`을 이용했습니다.  

KNN 모델의 특징은 비모수적 방법을 이용하며,  
높은 정확도를 가지지만 cost가 많이 든다는 단점이 있습니다.

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

로지스틱 회귀모델의 가장 좋은 성능은 **0.7433** 이며, KNN 모델의 가장 좋은 성능은 **0.5983** 으로 나왔습니다.  
가장 성능이 좋은 로지스틱 회귀모델로 학습을 진행했습니다.  

```python
#가장 좋은 모델 설정
model = estimator_results[0].best_estimator_
model.fit(X_train, y_train)
```

이후 Validation set의 성능을 평가하고 정확도를 확인했습니다.  

```python
# validation 예측
pred = model.predict(X_vld)

# validation set 성능 확인
print(classification_report(y_vld, pred))
print(confusion_matrix(y_vld, pred))
```




## 3. 딥러닝

### 탐색적 자료분석

### 모델링



## 4. 정리
