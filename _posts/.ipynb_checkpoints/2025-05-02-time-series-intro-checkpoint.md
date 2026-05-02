---
title: "시계열 분석 입문: Stationary vs Non-Stationary"
date: 2025-05-02 09:00:00 +0900
categories: [Time Series, 기초]
tags: [python, pandas, statsmodels, ADF-test]
math: true                    # 수식 사용 시
---

## 서론
*테스트 포스트 입니다*
시계열 데이터란 시간 순서대로 관측된 데이터입니다...

## 정상성(Stationarity) 검정

ADF 검정 코드:

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['value'])
print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')
```

## 수식 예시

$$y_t = \phi_1 y_{t-1} + \epsilon_t$$