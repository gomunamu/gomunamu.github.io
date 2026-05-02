---

title: "시계열 분석 입문: Stationary vs Non-Stationary"
date: 2026-05-03 09:00:00 +0900
categories: [Time Series, 기초]
tags: [python, pandas, statsmodels, ADF-test, stationarity, unit-root]

math: true
---

\---

## 서론

시계열 분석을 처음 공부하다 보면 "정상성(Stationarity)"이라는 단어를 피할 수 없습니다.  
ARIMA, VAR, GARCH 등 고전적인 시계열 모델들의 이론적 기반을 들여다보면, 이 모델들은 모두 **데이터가 정상적(stationary)이다**라는 가정 위에 설계되어 있습니다.

이 포스팅에서는 아래 질문들을 순서대로 풀어갑니다.

1. 정상성이란 무엇인가?
2. 왜 비정상 시계열로는 분석/예측이 어려운가?
3. 강정상성 vs 약정상성 — 실무에서 어느 쪽을 쓰는가?
4. 단위근(Unit Root)이란 무엇인가?
5. 시계열을 구성하는 요소는 무엇인가?
6. 어떤 요소가 비정상성을 유발하며, 어떻게 정상화하는가?

\---

## 1\. 정상성(Stationarity)이란?

직관적으로 말하면, **시계열의 통계적 특성이 시간이 흘러도 변하지 않는 상태**입니다.

"통계적 특성"은 크게 세 가지를 의미합니다.

|특성|수식|의미|
|-|-|-|
|평균|$E\[y\_t] = \\mu$|시간에 무관하게 일정|
|분산|$\\text{Var}(y\_t) = \\sigma^2 < \\infty$|시간에 무관하게 유한·일정|
|자기공분산|$\\text{Cov}(y\_t, y\_{t-k}) = \\gamma(k)$|시차(lag) $k$에만 의존, $t$에는 무관|

### 1.1 강정상성(Strict Stationarity)

임의의 시점 집합 ${t\_1, \\ldots, t\_n}$에 대해, **결합 분포** 자체가 시간 이동에 불변인 경우입니다.

$$
(y\_{t\_1}, \\ldots, y\_{t\_n}) \\overset{d}{=} (y\_{t\_1+h}, \\ldots, y\_{t\_n+h}), \\quad \\forall h
$$

분포의 **모든 모멘트**가 불변이어야 하므로 매우 강한 조건입니다. 이론적으로 깔끔하지만, 실제 데이터에서 이를 검증하는 것은 사실상 불가능합니다.

### 1.2 약정상성(Weak / Covariance Stationarity)

실무와 대부분의 교과서에서 "정상성"이라 함은 사실상 **약정상성**을 의미합니다.

* $E\[y\_t] = \\mu$ (일정한 평균)
* $E\[y\_t^2] < \\infty$ (유한 분산)
* $\\text{Cov}(y\_t, y\_{t-k}) = \\gamma(k)$ (자기공분산이 시차에만 의존)

분포 전체가 아닌 **1·2차 모멘트**만 불변이면 됩니다. 정규분포를 가정하면 강/약정상성이 동치가 되므로, 대부분의 모수적 시계열 모델은 약정상성으로 충분합니다.

> \*\*실무 요약:\*\* ARIMA, SARIMA, VAR 등 고전 모델에서 요구하는 정상성은 \*\*약정상성\*\*입니다.

\---

## 2\. 왜 비정상 시계열은 분석/예측이 어려운가?

비정상 시계열을 그대로 고전 모델에 투입했을 때 발생하는 문제는 크게 두 가지입니다.

### 2.1 허구적 회귀(Spurious Regression)

서로 무관한 두 비정상 시계열 $y\_t$와 $x\_t$를 단순 회귀하면, 실제로는 아무 관계가 없어도 $R^2$가 높고 t-통계량이 유의하게 나타납니다. Granger \& Newbold(1974)가 실험적으로 보였고, Phillips(1986)가 이론적으로 증명한 현상입니다.

추정량이 \*\*일치성(consistency)\*\*을 잃고, 검정통계량이 표준 분포를 따르지 않아서 p-value 자체가 무의미해집니다.

### 2.2 파라미터 추정의 불안정성

ARMA 모델은 정상성 조건 아래서만 MLE·OLS 추정량이 일치성과 점근적 정규성을 가집니다. 비정상 데이터에서는 추정된 파라미터의 표준오차가 발산하거나 왜곡되어, 신뢰구간과 예측 구간 자체를 신뢰할 수 없게 됩니다.

즉, "정상성이 전제되어야 추정된 파라미터가 통계적으로 의미 있고, 예측 구간도 신뢰할 수 있다"는 것이 핵심입니다.

\---

## 3\. 단위근(Unit Root)

비정상성의 대표적 원인 중 하나가 **단위근**입니다.

AR(1) 모델 $y\_t = \\phi y\_{t-1} + \\epsilon\_t$를 생각해 봅시다.

* $|\\phi| < 1$: 정상 과정 (충격이 기하급수적으로 소멸)
* $\\phi = 1$: **단위근** → 랜덤워크, 충격이 영구적으로 누적
* $|\\phi| > 1$: 폭발적 과정 (발산)

$\\phi = 1$인 경우를 특성방정식 $1 - \\phi L = 0$의 근이 단위원(unit circle) 위에 있다고 해서 **단위근**이라 부릅니다. 이 경우 분산이 $t$에 비례하여 증가하므로 약정상성 조건을 위반합니다.

$$
\\text{Var}(y\_t) = t\\sigma^2 \\to \\infty \\quad (t \\to \\infty)
$$

### 3.1 ADF 검정 (Augmented Dickey-Fuller Test)

단위근 존재 여부를 검정하는 가장 보편적인 방법입니다.

* $H\_0$: 단위근이 존재한다 (비정상)
* $H\_1$: 단위근이 없다 (정상)

```python
from statsmodels.tsa.stattools import adfuller
import pandas as pd

def run\_adf(series: pd.Series, label: str = "Series") -> None:
    result = adfuller(series.dropna(), autolag="AIC")
    print(f"\[{label}]")
    print(f"  ADF Statistic : {result\[0]:.4f}")
    print(f"  p-value       : {result\[1]:.4f}")
    for key, val in result\[4].items():
        print(f"  Critical ({key}): {val:.4f}")
    conclusion = "비정상 (단위근 존재)" if result\[1] > 0.05 else "정상"
    print(f"  판정           : {conclusion}\\n")
```

> \*\*주의:\*\* ADF 검정은 귀무가설이 "비정상"이므로, p-value > 0.05이면 비정상을 기각하지 못합니다. 단위근 검정은 \*\*검정력(power)\*\*이 낮고 구조적 변화에 취약하므로, KPSS 검정(귀무가설: 정상)과 함께 사용하면 더 견고한 판단이 가능합니다.

\---

## 4\. 시계열의 구성 요소

시계열 $y\_t$는 일반적으로 아래 네 가지 요소로 분해됩니다.

$$
y\_t = T\_t + S\_t + C\_t + R\_t
$$

|기호|이름|설명|
|-|-|-|
|$T\_t$|추세(Trend)|장기적으로 증가/감소하는 방향성|
|$S\_t$|계절성(Seasonality)|일정 주기로 반복되는 패턴 (주, 월, 연)|
|$C\_t$|순환(Cycle)|경기 사이클 같은 장기 비주기 변동|
|$R\_t$|잔차(Residual/Irregular)|위 셋으로 설명되지 않는 불규칙 성분|

실무에서는 순환 성분을 추세에 흡수시켜 $T\_t + C\_t$로 묶는 경우가 많습니다.

분해 방식은 두 가지입니다.

* **가법 모형:** $y\_t = T\_t + S\_t + R\_t$ (계절 변동 폭이 일정할 때)
* **승법 모형:** $y\_t = T\_t \\times S\_t \\times R\_t$ (변동 폭이 수준에 비례할 때 → 로그 변환으로 가법화 가능)

\---

## 5\. 어떤 성분이 비정상성을 유발하며, 어떻게 정상화하는가?

### 5.1 추세(Trend) → 차분 또는 추세 제거

추세는 **평균이 시간에 따라 변화**하게 만들어 정상성을 위반합니다.

**결정론적 추세(Deterministic Trend):** 회귀로 추세를 제거(Trend-Stationary, TS 과정)

```python
import numpy as np
from statsmodels.regression.linear\_model import OLS
import statsmodels.api as sm

t = np.arange(len(series))
X = sm.add\_constant(t)
model = OLS(series, X).fit()
detrended = series - model.fittedvalues
```

**확률적 추세(Stochastic Trend, 단위근):** 차분(Differencing)으로 제거 (Difference-Stationary, DS 과정)

$$
\\Delta y\_t = y\_t - y\_{t-1}
$$

```python
diff\_series = series.diff().dropna()
```

> \*\*DS vs TS 구분이 중요한 이유:\*\* TS 과정에 차분을 적용하면 과차분(overdifferencing)이 발생하고, DS 과정에 추세 제거만 하면 여전히 비정상입니다. ADF/KPSS 검정으로 판단하는 것이 원칙입니다.

### 5.2 계절성(Seasonality) → 계절 차분 또는 계절 조정

계절성은 **주기적으로 평균이 이동**하므로 정상성을 위반합니다.

계절 주기 $m$에 대한 계절 차분:

$$
\\Delta\_m y\_t = y\_t - y\_{t-m}
$$

```python
m = 12  # 월별 데이터
seasonal\_diff = series.diff(m).dropna()
```

### 5.3 분산의 비균일성(Heteroscedasticity) → 변환

분산이 시간에 따라 변하면 약정상성의 분산 조건을 위반합니다.

* **로그 변환:** $\\log(y\_t)$ — 지수적으로 증가하는 분산에 효과적
* **Box-Cox 변환:** 로그를 일반화한 형태

```python
from scipy.stats import boxcox

transformed, lam = boxcox(series\[series > 0])
print(f"최적 λ: {lam:.4f}")
# λ ≈ 0 이면 로그 변환과 동일
```

### 5.4 정리: 비정상 원인과 처방

|비정상 원인|정상화 방법|
|-|-|
|결정론적 추세|추세 회귀 후 잔차 사용 (Detrending)|
|확률적 추세 (단위근)|1차 차분 $\\Delta y\_t$|
|계절성|계절 차분 $\\Delta\_m y\_t$|
|이분산성|로그·Box-Cox 변환|
|추세 + 계절성 복합|변환 후 차분, 계절 차분 복합 적용|

\---

## 6\. 종합 예시: 단계별 정상화 파이프라인

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal\_decompose

def check\_stationarity(series: pd.Series, label: str = "") -> None:
    """ADF + KPSS 이중 검정"""
    adf = adfuller(series.dropna(), autolag="AIC")
    kpss\_stat, kpss\_p, \_, \_ = kpss(series.dropna(), regression="c", nlags="auto")

    print(f"=== {label} ===")
    print(f"ADF  p-value : {adf\[1]:.4f}  →  {'정상' if adf\[1] < 0.05 else '비정상'}")
    print(f"KPSS p-value : {kpss\_p:.4f}  →  {'정상' if kpss\_p > 0.05 else '비정상'}")
    print()

# 예시 데이터 (AirPassengers 스타일)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read\_csv(url, index\_col=0, parse\_dates=True)
series = df.squeeze()

# Step 1: 원본 검정
check\_stationarity(series, "원본")

# Step 2: 로그 변환 (이분산 완화)
log\_series = series.apply("log")  # np.log(series)
check\_stationarity(log\_series, "로그 변환")

# Step 3: 계절 차분 (m=12)
seasonal\_diff = log\_series.diff(12).dropna()
check\_stationarity(seasonal\_diff, "로그 + 계절차분")

# Step 4: 1차 차분 추가
final = seasonal\_diff.diff().dropna()
check\_stationarity(final, "로그 + 계절차분 + 1차차분")
```

\---

## 마치며

정상성은 단순히 "이 데이터가 좋은 데이터냐"의 문제가 아닙니다.  
**고전 시계열 모델이 통계적으로 의미 있는 결과를 내려면 반드시 전제되어야 하는 입력 조건**이며, 이를 어기면 파라미터 추정과 예측 구간 전체가 신뢰를 잃습니다.

비정상성의 원인을 진단하고 — 추세인지, 계절성인지, 이분산인지, 단위근인지 — 그에 맞는 처방을 내리는 것이 시계열 분석의 첫 번째 단계입니다.

다음 편에서는 정상화된 시계열에 ARMA 모형을 적합하는 과정, 즉 \*\*모수 추정과 모형 선택(AIC/BIC 기반 차수 결정)\*\*을 다루겠습니다.

\---

### 참고문헌

* Box, G. E. P., Jenkins, G. M., Reinsel, G. C., \& Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
* Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
* Granger, C. W. J., \& Newbold, P. (1974). Spurious regressions in econometrics. *Journal of Econometrics*, 2(2), 111–120.
* Phillips, P. C. B. (1986). Understanding spurious regressions in econometrics. *Journal of Econometrics*, 33(3), 311–340.
* Dickey, D. A., \& Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association*, 74(366), 427–431.
* Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., \& Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. *Journal of Econometrics*, 54(1–3), 159–178.

