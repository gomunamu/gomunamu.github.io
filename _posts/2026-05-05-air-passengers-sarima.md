\---

title: "Air Passengers로 SARIMA 끝까지 가기 — 진단부터 예측까지"

date: 2026-05-05 09:00:00 +0900

categories: \[Time Series]

tags: \[sarima, arima, box-jenkins, air-passengers, forecast, residual-diagnostics, time-series]

math: true

\---



지금까지의 시리즈에서 정상성, 모수 추정, AR/MA/ARMA/ARIMA/SARIMA를 차례로 다뤘습니다. 이번 글은 그 모든 것을 \*\*실제 데이터 한 묶음에 적용해 처음부터 끝까지 한 사이클을 완주\*\*하는 글입니다.



데이터는 시계열 분야의 고전 — \*\*Air Passengers\*\* 입니다. 1949년 1월부터 1960년 12월까지 월별 국제선 항공 여객 수(천 명 단위, 144개 데이터)이고, Box \& Jenkins(1976)의 \*Time Series Analysis\*에서 "Series G"로 등장한 이래 시계열 교과서의 표준 예제로 쓰여 왔습니다. 추세, 곱셈형 계절성, 분산 변화가 모두 들어 있어 SARIMA의 모든 절차를 한 번에 보여 줄 수 있습니다.



전체 절차는 7단계입니다.



1\. 데이터 첫 인상과 분해

2\. 변환 결정 — 로그를 씌울지 말지

3\. 정상성 진단과 차분

4\. 차수 식별 — ACF/PACF와 auto\_arima

5\. 적합과 summary 해독

6\. 잔차 진단

7\. 예측과 평가



각 단계마다 코드, 결과, 그리고 흔히 빠지는 함정을 짚겠습니다.



\## 1. 데이터 첫 인상



먼저 데이터를 불러서 그려 봅니다. 시계열 분석의 진단은 언제나 \*\*눈으로 보는 것\*\*부터 시작입니다.



```python

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal\_decompose



\# Air Passengers 데이터 (Box-Jenkins Series G, 1949-1960 월별)

\# CSV로 받았다고 가정하고 진행

series = pd.read\_csv('air\_passengers.csv', index\_col=0, parse\_dates=True).squeeze()

series.plot(figsize=(11, 3))

```



그리고 곱셈형(multiplicative) 분해를 한 번 해 봅니다. Air Passengers는 시간이 지날수록 \*\*계절 변동의 폭 자체가 커지는\*\* 모습이라 가법 모형보다 곱셈 모형이 더 잘 맞습니다.



```python

dec = seasonal\_decompose(series, model='multiplicative', period=12)

```



!\[데이터 분해](/assets/img/posts/fig\_p1\_decompose.png)



그림에서 세 가지가 한눈에 들어옵니다.



\- \*\*Trend (두 번째 패널)\*\*: 우상향 추세가 분명. 1949년 \~120명에서 1960년 \~470명으로 거의 4배 성장.

\- \*\*Seasonal (세 번째 패널)\*\*: 매년 동일한 패턴. 여름(7\~8월)이 가장 높고 겨울이 낮음. 곱셈형이라 1.0 주변에서 진동.

\- \*\*Residual (네 번째 패널)\*\*: 추세와 계절성을 빼고 남은 부분. 1.0 주변에서 흔들리고 있어 모형이 데이터를 잘 분해했음을 보여줌.



> \*\*함정 박스 1 — 분해 모형 선택\*\*

> 

> `seasonal\_decompose`의 `model` 인자에는 `'additive'`와 `'multiplicative'`가 있습니다. \*\*계절 변동의 폭이 시간에 따라 일정하면 가법, 변동의 폭이 수준에 비례해 커지면 곱셈\*\*입니다. 기준이 헷갈리면 그림을 두 가지로 다 그려 보고 잔차가 더 작고 패턴 없는 쪽을 고르면 됩니다. Air Passengers는 명백히 곱셈입니다.



\## 2. 변환 결정 — 로그가 필요한가



곱셈형 데이터를 가법형으로 바꾸는 가장 표준적인 방법은 \*\*로그 변환\*\*입니다. $y\_t = T\_t \\times S\_t \\times R\_t$에 로그를 씌우면 $\\log y\_t = \\log T\_t + \\log S\_t + \\log R\_t$가 되어 가법 구조가 됩니다.



원본과 로그 변환 후를 비교해 봅니다.



```python

log\_s = np.log(series)

```



!\[로그 변환 효과](/assets/img/posts/fig\_p2\_logtransform.png)



위 패널의 원본을 보면 1949\~1950년대 초의 계절 변동 폭은 $\\pm 30$명 정도인데, 1960년 즈음에는 $\\pm 100$명 이상으로 벌어집니다. 분산이 시간에 따라 커지는 것이지요. 약정상성의 두 번째 조건(분산이 시간에 무관)을 정면으로 위반합니다.



아래 패널의 로그 변환 후는 이 문제가 사라졌습니다. 1949년의 진폭과 1960년의 진폭이 비슷합니다. \*\*이제부터는 $\\log y\_t$를 다루겠습니다.\*\*



> \*\*함정 박스 2 — 차분만 해서는 분산 문제가 해결되지 않는다\*\*

> 

> 차분은 평균 비정상성(추세)을 다스리는 도구지, 분산 비정상성(이분산)을 잡지 않습니다. 분산이 시간에 따라 커지는 데이터에 차분만 적용하면 차분 후의 시계열도 분산이 여전히 시간에 따라 변합니다. \*\*분산 변화가 의심되면 차분 전에 로그/Box-Cox 변환을 먼저\*\* 적용하는 것이 표준 절차입니다.



\## 3. 정상성 진단과 차분



이제 $\\log y\_t$가 정상인지 검정합니다. ADF와 KPSS 두 검정을 함께 봅니다. (ADF의 귀무가설은 "비정상", KPSS는 "정상"이라 둘 다 정상성을 지지할 때 결론이 견고합니다.)



```python

from statsmodels.tsa.stattools import adfuller, kpss



def stationarity\_check(s, label):

&#x20;   adf\_p = adfuller(s.dropna(), autolag='AIC')\[1]

&#x20;   kpss\_p = kpss(s.dropna(), regression='c', nlags='auto')\[1]

&#x20;   print(f"{label:30s} ADF p={adf\_p:.4f}, KPSS p={kpss\_p:.4f}")

```



여러 변환에 대해 검정을 돌려 본 결과는 다음과 같습니다.



| 변환 | ADF p-value | KPSS p-value | 판정 |

|---|---:|---:|---|

| 원본 $y\_t$ | 0.9919 | 0.0100 | 비정상 (둘 다) |

| $\\log y\_t$ | 0.4224 | 0.0100 | 비정상 (둘 다) |

| $\\log y\_t$ + 1차 차분 ($\\Delta$) | 0.0711 | 0.1000 | 애매 (KPSS는 OK, ADF는 borderline) |

| $\\log y\_t$ + 계절 차분 ($\\Delta\_{12}$) | 0.0724 | 0.1000 | 애매 (마찬가지) |

| $\\log y\_t$ + $\\Delta$ + $\\Delta\_{12}$ | \*\*0.0002\*\* | \*\*0.1000\*\* | \*\*정상 (양쪽 모두 합의)\*\* |



\*\*1차 차분과 계절 차분을 모두 적용한 후\*\*에야 두 검정이 정상성에 합의합니다. 시각적으로도 확인해 봅니다.



```python

log\_diff\_both = log\_s.diff().diff(12).dropna()

```



!\[차분 단계](/assets/img/posts/fig\_p3\_differencing.png)



세 번째 패널의 두 번 차분된 시계열이 0 주변에서 안정적으로 흔들리는 모습이 명확합니다. 



이로써 SARIMA 표기의 차분 차수가 결정됐습니다: \*\*$d = 1, D = 1, m = 12$\*\*.



\## 4. 차수 식별 — ACF/PACF와 auto\_arima



남은 것은 $p, q, P, Q$ 입니다. 두 번 차분된 시계열의 ACF와 PACF를 봅니다.



```python

from statsmodels.graphics.tsaplots import plot\_acf, plot\_pacf



fig, axes = plt.subplots(1, 2, figsize=(12, 4))

plot\_acf(log\_diff\_both, ax=axes\[0], lags=24)

plot\_pacf(log\_diff\_both, ax=axes\[1], lags=24, method='ywm')

```



!\[차분된 시계열의 ACF/PACF](/assets/img/posts/fig\_p4\_acf\_pacf.png)



읽는 법은 두 부분으로 나눕니다.



\*\*일반(non-seasonal) 부분 — 시차 1 근처\*\*:

\- ACF lag 1에서 큰 음수값(약 −0.34), 그 이후 빠르게 띠 안으로 → \*\*MA(1) 시그니처\*\*

\- PACF는 점진적으로 감쇠 → MA 모델임을 뒷받침

\- 따라서 $p = 0, q = 1$



\*\*계절(seasonal) 부분 — 시차 12 근처\*\*:

\- ACF lag 12에서 큰 음수값(약 −0.39) → \*\*계절 MA(1) 시그니처\*\*

\- PACF lag 12에서도 큰 음수값

\- 따라서 $P = 0, Q = 1$



ACF/PACF 만으로 도달한 후보가 \*\*SARIMA(0, 1, 1)(0, 1, 1)$\_{12}$\*\*입니다. 이게 Box \& Jenkins(1976)가 이 데이터에 처음 제시한 그 유명한 \*\*"airline model"\*\* 입니다.\[^1]



`auto\_arima`로도 확인해 봅니다.



```python

from pmdarima import auto\_arima



best = auto\_arima(

&#x20;   log\_s,

&#x20;   start\_p=0, max\_p=3,

&#x20;   start\_q=0, max\_q=3,

&#x20;   d=1, D=1, m=12,

&#x20;   seasonal=True,

&#x20;   start\_P=0, max\_P=2,

&#x20;   start\_Q=0, max\_Q=2,

&#x20;   information\_criterion='aic',

&#x20;   stepwise=True,

&#x20;   suppress\_warnings=True,

)

print(best.order, best.seasonal\_order)

\# → (0, 1, 1) (0, 1, 1, 12)

```



`auto\_arima`도 같은 모형을 골랐습니다. 이 데이터에서는 \*\*수동 식별과 자동 탐색이 일치하는 행복한 케이스\*\*입니다. 항상 이렇게 깔끔하지는 않습니다.



\## 5. 적합과 summary 해독



이제 모형을 적합합니다. \*\*단, 평가를 위해 마지막 24개월(1959-1960)은 테스트 셋으로 떼어 놓습니다.\*\*



```python

from statsmodels.tsa.statespace.sarimax import SARIMAX



train = log\_s.iloc\[:-24]   # 1949-01 \~ 1958-12 (120개월)

test  = log\_s.iloc\[-24:]   # 1959-01 \~ 1960-12 (24개월)



model = SARIMAX(train, order=(0, 1, 1), seasonal\_order=(0, 1, 1, 12))

result = model.fit(disp=False)

print(result.summary())

```



> \*\*함정 박스 3 — 시계열에서 무작위 train/test 분할은 데이터 누수\*\*

> 

> ML 일반에서 흔히 쓰는 `train\_test\_split(shuffle=True)`은 시계열에서 \*\*절대\*\* 쓰면 안 됩니다. 무작위로 섞으면 미래 시점이 학습 데이터에 들어가 모형이 본 적 없는 미래를 보고 학습한 셈이 되고, 그 결과 테스트 성능이 비현실적으로 좋게 나옵니다 (data leakage). 시계열은 \*\*반드시 시간 순서대로\*\* 마지막 일부를 잘라 테스트 셋으로 써야 합니다.



summary 출력의 핵심 부분을 보면:



```

&#x20;                                SARIMAX Results                                      

======================================================================================

Dep. Variable:                    passengers   No. Observations:                  120

Model:         SARIMAX(0, 1, 1)x(0, 1, 1, 12)   Log Likelihood                 197.505

&#x20;                                               AIC                           -389.010

&#x20;                                               BIC                           -380.991



==============================================================================

&#x20;                coef    std err          z      P>|z|      \[0.025      0.975]

\------------------------------------------------------------------------------

ma.L1         -0.3423      0.087     -3.920      0.000      -0.513      -0.171

ma.S.L12      -0.5405      0.105     -5.155      0.000      -0.746      -0.335

sigma2         0.0014      0.000      7.864      0.000       0.001       0.002

==============================================================================

```



읽는 포인트는 다음과 같습니다.



\- \*\*`ma.L1` = −0.3423\*\*: 일반 MA 계수 $\\theta\_1$. p < 0.001로 매우 유의.

\- \*\*`ma.S.L12` = −0.5405\*\*: 계절 MA 계수 $\\Theta\_1$. p < 0.001로 매우 유의.

\- \*\*`sigma2`\*\*: 잔차 분산.

\- \*\*AIC = −389.01, BIC = −380.99\*\*: 다른 모형과 비교할 때 사용.



두 모수 모두 통계적으로 유의하므로 모델 구조가 타당합니다. 이제 잔차를 봐야 합니다.



> \*\*함정 박스 4 — AIC가 작다고 끝난 게 아니다\*\*

> 

> AIC만 보고 모델을 채택하는 것은 위험합니다. AIC는 같은 데이터에 적합된 여러 모형을 비교하는 상대적 척도일 뿐, 절댓값에 의미가 없고 잔차의 품질도 보장하지 않습니다. \*\*AIC가 가장 작은 모형의 잔차에 자기상관이 남아 있다면 그 모형은 데이터를 충분히 설명하지 못한 것\*\*이고, 예측 신뢰구간도 잘못 계산됩니다. 다음 단계인 잔차 진단이 그래서 필수입니다.



\## 6. 잔차 진단



좋은 모형의 잔차는 \*\*백색잡음(white noise)\*\* 처럼 보여야 합니다. 즉 평균이 0이고, 분산이 일정하고, 시점 간 상관이 없으며, (정규성을 가정한 경우) 정규분포를 따릅니다. 이 네 가지를 한 번에 점검하는 4분할 그래프가 표준입니다.



```python

resid = result.resid\[14:]   # burn-in 구간 제거 (d=1, D=1, m=12 → 약 13개월)



fig, axes = plt.subplots(2, 2, figsize=(12, 8))

\# (좌상) 잔차 시계열, (우상) 히스토그램, (좌하) Q-Q, (우하) 잔차 ACF

```



!\[잔차 진단 4분할](/assets/img/posts/fig\_p5\_residuals.png)



각 패널을 한 줄씩 읽으면:



\- \*\*잔차 시계열 (좌상)\*\*: 0 주변에서 일정한 폭으로 흔들림. 추세 흔적, 분산 변화, 큰 outlier 없음. 좋음.

\- \*\*히스토그램 + 정규분포 (우상)\*\*: 종 모양으로 정규분포에 잘 맞음.

\- \*\*Q-Q 플롯 (좌하)\*\*: 점들이 빨간 직선에 거의 붙어 있음. 정규성 양호.

\- \*\*잔차 ACF (우하)\*\*: \*\*모든 막대가 95% 신뢰띠 안.\*\* 자기상관 없음. 가장 중요한 통과 신호.



수치 검정으로도 확인합니다.



```python

from statsmodels.stats.diagnostic import acorr\_ljungbox



acorr\_ljungbox(resid, lags=\[10, 20], return\_df=True)

```



| lag | lb\_stat | lb\_pvalue |

|---|---:|---:|

| 10 | 0.22 | 1.00 |

| 20 | 1.05 | 1.00 |



\*\*Ljung-Box 검정\*\*의 귀무가설은 "잔차에 자기상관 없음"입니다.\[^2] p-value가 1.00에 가까우면 자기상관 가설을 강력히 기각하지 못함, 즉 \*\*자기상관이 없다는 결론에 합의\*\*됩니다. 모형이 데이터의 시간 의존성을 충분히 흡수했다는 뜻입니다.



statsmodels의 SARIMAX summary에 자동으로 함께 출력되는 진단도 봅니다.



```

Ljung-Box (L1) (Q):                   0.01   Jarque-Bera (JB):                 0.94

Prob(Q):                              0.92   Prob(JB):                         0.63

Heteroskedasticity (H):               0.37   Skew:                             0.12

Prob(H) (two-sided):                  0.00   Kurtosis:                         3.39

```



\- \*\*Ljung-Box Prob(Q) = 0.92\*\*: 자기상관 없음. 통과.

\- \*\*Jarque-Bera Prob(JB) = 0.63\*\*: 정규성 가정 통과.

\- \*\*Heteroskedasticity Prob(H) = 0.00\*\*: ⚠️ \*\*이분산성이 있다\*\*는 신호. 잔차의 분산이 시간에 따라 살짝 변할 수 있음을 시사. 다만 핵심 진단(자기상관 없음, 정규성)은 통과했고, 로그 변환으로 대부분의 분산 문제는 해결한 상태라 실용적 사용에는 무리가 없는 수준입니다. 더 엄밀한 분석이 필요하다면 GARCH로 분산 모형을 추가하는 방향이 있습니다.



종합하면 \*\*잔차 진단을 양호하게 통과\*\*했습니다. 모형을 그대로 예측에 사용해도 됩니다.



\## 7. 예측과 평가



테스트 셋(마지막 24개월)에 대해 예측을 수행합니다.



```python

forecast = result.get\_forecast(steps=24)

pred\_log = forecast.predicted\_mean         # 로그 스케일 예측

pred\_ci  = forecast.conf\_int(alpha=0.05)   # 95% 신뢰구간



\# 원래 스케일로 되돌림 (지수)

pred = np.exp(pred\_log)

ci\_low  = np.exp(pred\_ci.iloc\[:, 0])

ci\_high = np.exp(pred\_ci.iloc\[:, 1])

test\_actual = np.exp(test)

```



!\[24개월 예측](/assets/img/posts/fig\_p6\_forecast.png)



빨간 실선이 점예측, 분홍 띠가 95% 신뢰구간입니다. 점선의 실제값(검은 점선)이 신뢰구간 안에 거의 모두 들어와 있고, 추세와 계절성을 잘 따라가는 모습입니다.



신뢰구간이 시간이 갈수록 넓어지는 것을 주목하세요. 이건 \*\*예측 불확실성이 시간이 길어질수록 누적\*\*되기 때문입니다. SARIMA의 차분 구조($d = 1, D = 1$)가 누적 불확실성을 만들어 내는 메커니즘이고, 이는 모형의 본질적 성질입니다. 예측 수평선(forecast horizon)이 길수록 신뢰구간 폭이 발산하는 것은 정상입니다.



수치 평가:



```python

mae  = np.mean(np.abs(test\_actual - pred))

rmse = np.sqrt(np.mean((test\_actual - pred) \*\* 2))

mape = np.mean(np.abs((test\_actual - pred) / test\_actual)) \* 100

```



| 지표 | 값 |

|---|---:|

| MAE | 39.45 |

| RMSE | 43.19 |

| MAPE | \*\*8.52%\*\* |



월별 항공 여객 수 예측에서 \*\*MAPE 8.52%\*\*는 양호한 수준입니다. 24개월 같은 비교적 긴 horizon에서 이 정도 정확도면 실무적으로 사용 가능합니다.



> \*\*함정 박스 5 — 로그 스케일에서 평가하지 마라\*\*

> 

> 로그 변환된 시계열에서 직접 RMSE/MAE를 계산하면 결과가 원래 단위와 분리되어 해석이 어렵습니다. 또 로그 공간의 평균이 원 공간의 평균과 다르다는 통계적 문제도 있습니다. \*\*반드시 `np.exp()`로 원 스케일로 되돌린 다음\*\* 평가하시기 바랍니다.



\## 마치며 — 이번에 따라간 절차



7단계를 한 번에 정리하면:



| 단계 | 행한 일 | 결과 |

|---|---|---|

| 1 | 시각화와 분해 | 추세 + 곱셈형 계절성 + 분산 변화 확인 |

| 2 | 로그 변환 | 분산 안정화 |

| 3 | ADF/KPSS + 차분 | $d=1, D=1, m=12$ 결정 |

| 4 | ACF/PACF + auto\_arima | $(p,q,P,Q) = (0,1,0,1)$ |

| 5 | SARIMAX 적합 | 두 모수 모두 유의 |

| 6 | 잔차 진단 | 자기상관 없음, 정규성 OK |

| 7 | 24개월 예측 | MAPE 8.52% |



이 절차가 \*\*Box-Jenkins 식별 절차\*\*의 정석입니다. 다른 데이터에 적용하실 때도 같은 7단계를 따르시면 됩니다. 다만 다음 두 가지를 항상 기억하시면 좋겠습니다.



\- \*\*각 단계의 결과를 다음 단계의 입력으로 쓰는 흐름\*\*이지, 한 번에 자동화로 끝낼 수 있는 작업이 아닙니다. 사람이 매 단계 결과를 보고 판단해야 합니다.

\- \*\*모형이 잔차 진단을 통과해야 비로소 예측 신뢰구간이 신뢰할 만\*\*합니다. 점예측만 보면 안 됩니다.



이 글로 ARIMA/SARIMA 시리즈는 일단 한 묶음을 마무리합니다. 다음 시리즈로는 다음 중 하나를 생각하고 있습니다.



\- \*\*잔차 진단 깊이 들어가기\*\*: Ljung-Box, Jarque-Bera, ARCH 검정의 통계적 배경

\- \*\*베이지안 시계열\*\*: PyMC로 SARIMA에 사전분포를 입히기

\- \*\*머신러닝 시계열로 넘어가기\*\*: SARIMA가 잡지 못하는 비선형 패턴, LSTM/Temporal Fusion Transformer 등



\---



\[^1]: Box \& Jenkins(1976) \*Time Series Analysis\*가 이 데이터에 SARIMA(0,1,1)(0,1,1)$\_{12}$을 처음 제안하면서 "airline model"이라는 별명을 얻었습니다. 이후 시계열 입문 교재의 거의 모든 곳에서 표준 예제로 사용됩니다. 자세한 역사적 맥락은 Hyndman \& Athanasopoulos(2021) Section 9.9 참고.



\[^2]: Ljung-Box 검정 통계량은 $Q = n(n+2) \\sum\_{k=1}^{h} \\hat{\\rho}\_k^2 / (n-k)$로 정의되며, 귀무가설 하에서 자유도 $h - p - q$의 카이제곱 분포를 따릅니다. $h$는 검정에 사용한 시차의 개수, $p, q$는 ARMA 차수입니다. statsmodels의 SARIMAX summary에 자동 표시되는 "Ljung-Box (L1) (Q)"는 $h=1$의 결과입니다.



\---



\*\*참고문헌\*\*



\- Box, G. E. P., \& Jenkins, G. M. (1976). \*Time Series Analysis: Forecasting and Control\* (revised ed.). Holden-Day.

\- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., \& Ljung, G. M. (2015). \*Time Series Analysis: Forecasting and Control\* (5th ed.). Wiley.

\- Hyndman, R. J., \& Athanasopoulos, G. (2021). \*Forecasting: Principles and Practice\* (3rd ed.). OTexts. https://otexts.com/fpp3/

\- Ljung, G. M., \& Box, G. E. P. (1978). On a measure of lack of fit in time series models. \*Biometrika\*, 65(2), 297–303.



\---



AI의 도움을 받아 작성되었으며 최대한 레퍼런스를 밝히려 노력했으나 오류가 있을 수 있으니 정확한 정보를 다시 한번 확인하시기 바랍니다.



