---
title: "AR, MA, ARMA, ARIMA, SARIMA — 고전 시계열 모형 한 바퀴"
date: 2026-05-04 09:00:00 +0900
categories: [Time Series]
tags: [ar, ma, arma, arima, sarima, sarimax, acf, pacf, time-series]
math: true
---

지난 포스팅에서 정상성과 모수 추정의 발상을 정리했습니다. 이번 글에서는 그 위에 본격적인 모형들을 올려 보겠습니다. AR, MA, ARMA, ARIMA, SARIMA, 그리고 SARIMAX까지. 이름이 많아 보이지만 실은 **AR과 MA라는 두 가지 부품**을 어떻게 조립하느냐의 차이입니다.

이번 글의 흐름은 이렇습니다.

1. AR(1)과 MA(1) — 각각 한 시점 충격에 어떻게 반응하는가
2. 차수 p, q를 어떻게 결정하는가 — ACF/PACF의 역할
3. ARMA, ARIMA — 두 부품을 합치고, 차분으로 비정상을 다스리기
4. SARIMA — 계절성까지 다루기
5. SARIMAX — 외생변수를 추가하기 (그리고 다변량 시계열은 다른 이야기)

## 1. AR(1)과 MA(1) — 두 가지 부품

### 1.1 AR(1) — 자기회귀(AutoRegressive)

가장 단순한 자기회귀 모형 AR(1)은 다음과 같이 정의됩니다.

$$
y_t = \phi \, y_{t-1} + \epsilon_t
$$

여기서 $\epsilon_t$는 평균 0, 분산 $\sigma^2$의 백색잡음(white noise)입니다.[^1] **현재 값이 직전 값에 $\phi$배만큼 의존하고, 거기에 새 충격 $\epsilon_t$가 더해진다**는 구조입니다.

핵심 모수는 $\phi$ 하나입니다. $\phi$의 값이 시계열의 성격을 결정합니다.

- $\lvert \phi \rvert < 1$: 정상 과정
- $\phi = 1$: 단위근 (랜덤워크)
- $\lvert \phi \rvert > 1$: 폭발적 과정

이전 글에서 $\phi$ 값에 따라 시계열이 어떻게 달라지는지를 다루었으니, 여기서는 한 발 더 들어가서 **충격이 어떻게 전파되는가**를 보겠습니다.

### 1.2 MA(1) — 이동평균(Moving Average)

이름은 이동평균이지만, 일반적으로 말하는 "이동평균선"과는 다릅니다. **과거 충격들의 가중합**을 의미합니다.

$$
y_t = \epsilon_t + \theta \, \epsilon_{t-1}
$$

핵심 모수는 $\theta$ 하나입니다. **현재 값이 현재 충격 $\epsilon_t$와 직전 충격 $\epsilon_{t-1}$의 합으로 정의**되는 구조입니다.

여기서 헷갈리지 않게 짚고 가야 할 것이 있습니다. AR(1)에서는 "직전 **값** $y_{t-1}$"이 직접 들어가고, MA(1)에서는 "직전 **충격** $\epsilon_{t-1}$"이 들어갑니다. 이 차이가 두 모형의 행동 방식을 결정합니다.

### 1.3 한 시점만 충격을 가하면 어떻게 될까

이게 두 모형의 차이를 가장 극명하게 보여 주는 실험입니다. 모든 시점에서 $\epsilon_t = 0$이고, 오직 $t=10$에서만 $\epsilon_{10} = 5$인 상황을 생각해 봅시다. 그 후의 시계열을 따라가 보면:

![AR(1)과 MA(1)의 충격 반응](/assets/img/posts/fig_impulse_response.png)

**AR(1) (왼쪽, $\phi = 0.7$)**: $t=10$에서 5라는 충격을 맞고, 그 이후로는 충격이 0이지만 시계열은 $\phi = 0.7$의 비율로 곱해지며 천천히 0으로 수렴합니다. 직접 식을 풀어 보면:

$$
\begin{aligned}
y_{10} &= \phi \cdot 0 + 5 = 5 \\
y_{11} &= \phi \cdot 5 + 0 = 3.5 \\
y_{12} &= \phi \cdot 3.5 + 0 = 2.45 \\
y_{13} &= \phi \cdot 2.45 + 0 \approx 1.72 \\
&\vdots
\end{aligned}
$$

일반화하면 충격 후 $k$시점이 지나면 $y_{10+k} = \phi^k \cdot 5$입니다. 충격이 **무한히 길게, 기하급수적으로 감쇠하며 살아남는다**는 것이 AR의 성격입니다.

**MA(1) (오른쪽, $\theta = 0.7$)**: 같은 충격을 받으면 정확히 두 시점에만 영향이 나타나고 끝납니다.

$$
\begin{aligned}
y_{10} &= \epsilon_{10} + \theta \cdot \epsilon_{9} = 5 + 0 = 5 \\
y_{11} &= \epsilon_{11} + \theta \cdot \epsilon_{10} = 0 + 0.7 \cdot 5 = 3.5 \\
y_{12} &= \epsilon_{12} + \theta \cdot \epsilon_{11} = 0 + 0 = 0 \\
y_{13} &= 0
\end{aligned}
$$

MA(1)은 충격이 **딱 한 스텝만 살아남고 끝**납니다. MA(q)로 일반화하면 정확히 $q$ 스텝까지만 영향이 갑니다.

이 차이가 본질입니다.

> **한 줄 요약**: AR은 충격이 끝없이 감쇠하며 흐른다. MA는 정해진 차수만큼만 영향을 끼치고 끝난다.

## 2. 차수 결정 — ACF와 PACF

AR과 MA를 일반화하면 AR(p), MA(q)가 됩니다. p와 q를 어떻게 결정할까요? 이게 모형 선택의 출발점입니다.

두 도구를 사용합니다.

- **ACF (AutoCorrelation Function, 자기상관함수)**: 시점 $t$의 값과 시점 $t-k$의 값 사이의 상관계수. $k$는 시차(lag).
- **PACF (Partial AutoCorrelation Function, 편자기상관함수)**: 중간 시차들의 영향을 통제한 뒤의 순수한 시차 $k$ 상관.[^2]

각 시차마다 상관계수를 계산해서 막대그래프로 그리는 게 보통입니다. statsmodels로 한 줄이면 됩니다.

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(series, lags=30)
plot_pacf(series, lags=30)
```

### 2.1 차수 결정 규칙

이론적으로는 다음과 같은 패턴이 나옵니다.[^3]

| 모형 | ACF | PACF |
|---|---|---|
| **AR(p)** | 점진적으로 감쇠 (지수/사인) | 시차 $p$에서 절단 |
| **MA(q)** | 시차 $q$에서 절단 | 점진적으로 감쇠 |
| **ARMA(p,q)** | 점진적으로 감쇠 | 점진적으로 감쇠 |

"절단(cut-off)"이란 그 시차를 넘으면 막대 길이가 신뢰구간 안으로 떨어지는 것을 말합니다.

이 규칙이 작동하는 직관은 1.3절의 충격 반응에서 이미 보셨습니다.

- **MA(q)**는 충격이 정확히 q 시점까지만 영향이 가니까, $y_t$와 $y_{t-k}$의 상관(ACF)이 $k > q$에서 0이 됩니다 → ACF cut-off at q.
- **AR(p)**는 충격이 무한히 감쇠하니까 ACF는 절대 0이 안 되고 점진적으로 줄어듭니다. 대신 PACF는 시차 $p$에서 끊깁니다.

### 2.2 현실은 그리 깔끔하지 않다

여기까지는 교과서적입니다. 실제 데이터에서는 ACF와 PACF가 둘 다 애매하게 감쇠하거나, 잡음이 많아 패턴이 잘 안 보일 때가 흔합니다. 그래서 실무에서는 ACF/PACF로 후보 차수 몇 개를 추리고 → 각 후보를 적합한 다음 → **AIC, BIC 같은 정보 기준(information criterion)** 으로 최종 선택합니다.[^4]

이번 글에서는 차수 결정을 그 정도까지만 다루고, 차수 $p, q$가 정해졌다고 가정한 상태로 모형을 쌓아 가겠습니다.

## 3. ARMA — 두 부품 합치기

ARMA(p, q)는 단순히 AR과 MA를 더한 것입니다.

$$
y_t = \underbrace{\phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p}}_{\text{AR(p)}} + \underbrace{\epsilon_t + \theta_1 \epsilon_{t-1} + \cdots + \theta_q \epsilon_{t-q}}_{\text{MA(q)}}
$$

모수는 $\phi_1, \ldots, \phi_p$ ($p$개)와 $\theta_1, \ldots, \theta_q$ ($q$개)로 총 $p+q$개입니다 (오차 분산 $\sigma^2$ 제외).

ARMA의 강력함은 **하나의 시계열에 자기회귀 성분과 이동평균 성분이 동시에 있을 때** 둘 다 잡아 준다는 점입니다. AR만 쓰면 MA적 성분이 잔차에 남고, MA만 쓰면 AR적 성분이 남습니다.

다만 **ARMA는 약정상 시계열을 가정**합니다. 평균과 분산이 시간에 따라 변하지 않아야 한다는 조건입니다. 추세가 있는 시계열, 즉 비정상 시계열에는 그대로 적용할 수 없습니다.

## 4. ARIMA — 차분으로 비정상을 정상화

비정상 시계열을 어떻게 다룰까요? 이전 글에서 다룬 **차분(differencing)** 이 답입니다. 차분을 한 번 적용하면 추세가 제거되고, 두 번 적용하면 곡률 있는 추세까지 제거됩니다.

ARIMA(p, d, q)는 정확히 이런 모형입니다.

> **원본 시계열에 $d$번 차분을 적용한 결과를 ARMA(p, q)로 적합한다.**

수식으로는 차분 연산자 $\Delta y_t = y_t - y_{t-1}$를 써서:

$$
\Delta^d y_t \sim \text{ARMA}(p, q)
$$

세 모수의 의미는 다음과 같습니다.

- **$p$**: AR 차수
- **$d$**: 차분 횟수
- **$q$**: MA 차수

### 4.1 d는 어떻게 정하는가

$d$는 통계적 추정이라기보다 **진단 기반 결정**입니다. 절차는 이렇습니다.

1. 원본 시계열에 ADF/KPSS 검정 → 정상이면 $d=0$
2. 비정상이면 1차 차분 후 다시 검정 → 정상이면 $d=1$
3. 그래도 비정상이면 2차 차분 → $d=2$

**실무에서 $d \geq 3$인 경우는 거의 없습니다.** Hyndman & Athanasopoulos(2021)도 명시적으로 "거의 항상 $d=0, 1, 2$ 안에서 결정된다"고 지적합니다.[^5] 3차 이상의 차분이 필요하다는 건 시계열에 비정상성 외의 구조적 문제(예: 분산 비균일성, 구조변화)가 있다는 신호일 가능성이 높습니다. 이때는 차분 대신 로그/Box-Cox 변환이나 구조변화 모형을 먼저 검토하는 게 맞습니다.

### 4.2 정리

ARIMA(p, d, q)는 결국 다음의 조합입니다.

- **차분**으로 비정상 시계열을 정상으로 만들고
- **AR**로 자기회귀 성분을 잡고
- **MA**로 충격의 단기 영향을 잡는다

## 5. SARIMA — 계절성까지

월별 매출, 일별 전기 사용량, 시간대별 웹 트래픽 — 현실의 시계열은 대부분 **계절성(seasonality)** 을 가지고 있습니다. 매년 12월에 매출이 오르고, 매주 월요일에 트래픽이 급등하는 식이지요.

ARIMA만으로는 이런 주기적 패턴을 잡기 어렵습니다. 그래서 등장하는 것이 **SARIMA(Seasonal ARIMA)** 입니다.

### 5.1 SARIMA(p, d, q)(P, D, Q, m)

표기가 한 단계 길어졌습니다. 풀어 보면:

- $(p, d, q)$ — 일반(non-seasonal) 부분의 AR, 차분, MA 차수
- $(P, D, Q, m)$ — 계절(seasonal) 부분의 AR, 차분, MA 차수, 그리고 계절 주기 $m$

**$m$**(계절 주기)이 핵심입니다. 월별 데이터는 $m=12$, 일별이지만 주 단위 패턴이 있다면 $m=7$, 시간별이지만 일 단위 패턴이라면 $m=24$ 식입니다.

### 5.2 계절 차분이란

일반 차분이 $\Delta y_t = y_t - y_{t-1}$이라면, 계절 차분은 $\Delta_m y_t = y_t - y_{t-m}$입니다. **한 주기 전 같은 시점의 값을 빼는 것**입니다. 1년 전 같은 달의 값을 빼면 계절 효과가 사라집니다.

| 변환 | 정의 | 제거하는 성분 |
|---|---|---|
| 1차 차분 | $\Delta y_t = y_t - y_{t-1}$ | 추세 |
| 계절 차분 | $\Delta_m y_t = y_t - y_{t-m}$ | 계절성 |
| 둘 다 | $\Delta \Delta_m y_t$ | 추세 + 계절성 |

추세와 계절성이 모두 있는 시계열은 보통 두 가지 차분을 다 적용합니다.

### 5.3 시각화

추세와 계절성(주기 12)을 모두 가진 인공 시계열에 각 차분을 단계적으로 적용해 봤습니다.

![계절 차분의 효과](/assets/img/posts/fig_seasonal_diff.png)

위에서부터:

- **원본**: 우상향 추세에 12개월 주기 사인파가 얹혀 있음
- **1차 차분 후**: 추세는 사라졌지만 계절성은 그대로
- **계절 차분 후**: 계절성은 사라졌지만 추세 흔적이 남음 (값들이 0이 아닌 양수 영역에 모여 있음)
- **둘 다 적용**: 0 주변에서 안정적으로 흔들림 — 정상에 가까움

이 마지막 시리즈에 ARMA를 적합하면 SARIMA(p, 1, q)(P, 1, Q, 12) 형태가 됩니다.

### 5.4 차수가 많아 보이지만

SARIMA(p, d, q)(P, D, Q, m)는 모수 7개로 보이지만, 실제로 $m$은 데이터 특성상 정해지고($m=12$인지 $m=7$인지는 데이터를 보면 압니다), $d$와 $D$는 보통 0이나 1입니다. 핵심적으로 결정해야 하는 것은 $p, q, P, Q$ 네 개입니다.

실무에서는 `auto_arima`(pmdarima 패키지)나 R의 `forecast::auto.arima` 같은 자동 차수 선택 도구를 자주 씁니다. AIC/BIC 기반 그리드 탐색을 자동으로 해 줍니다.

## 6. SARIMAX — 외생변수까지

마지막 글자 X는 **eXogenous(외생변수)** 의 약자입니다.[^6] 다변량 시계열 모델이 아니라는 점을 분명히 짚고 가야 합니다.

### 6.1 외생변수란

예측 대상 시계열 $y_t$ 외에, **예측에 도움이 될 만한 다른 변수** $x_t$를 추가로 모델에 넣는 것입니다. 예를 들어:

- 매출($y_t$) 예측에 광고비($x_{1,t}$), 휴일 여부($x_{2,t}$)를 추가
- 전기 사용량($y_t$) 예측에 기온($x_{1,t}$), 요일($x_{2,t}$)을 추가

수식으로는 SARIMA에 외생변수 항이 더해진 형태입니다.

$$
y_t = \beta_1 x_{1,t} + \beta_2 x_{2,t} + \cdots + \text{SARIMA 부분}
$$

statsmodels의 `SARIMAX` 클래스에서 `exog` 인자로 외생변수를 넘기면 됩니다.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(y, exog=X, order=(1,1,1), seasonal_order=(1,1,1,12))
result = model.fit()
```

### 6.2 다변량 시계열은 다른 이야기

여기서 분명히 짚어 두어야 할 것이 있습니다. **SARIMAX는 진짜 다변량 시계열 모형이 아닙니다.** 외생변수 $x_t$는 "예측 대상이 아닌 입력 변수"로 취급되며, $x_t$의 미래 값은 별도로 알거나 가정해야 합니다.

여러 시계열을 **상호 영향**을 가진 채로 동시에 모델링하려면 다른 모형이 필요합니다.

- **VAR (Vector AutoRegression)**: 여러 시계열이 서로의 과거에 영향받는 다변량 자기회귀 모형
- **VECM (Vector Error Correction Model)**: 공적분(cointegration) 관계가 있는 다변량 비정상 시계열용
- **DFM (Dynamic Factor Model)**: 많은 시계열이 소수의 공통 요인에 의해 움직인다고 가정하는 모형

이 영역으로 들어가면 본격적인 다변량 통계학과 행렬대수가 필요해지고, 최근에는 머신러닝/딥러닝 기반 다변량 시계열 모형(LSTM, Temporal Fusion Transformer 등)도 활발히 사용됩니다. 이건 별도 시리즈로 다루는 게 맞을 것 같습니다.

## 7. 정리

| 모형 | 식 핵심 | 다루는 것 |
|---|---|---|
| **AR(p)** | $y_t = \sum \phi_i y_{t-i} + \epsilon_t$ | 자기회귀 |
| **MA(q)** | $y_t = \epsilon_t + \sum \theta_j \epsilon_{t-j}$ | 충격의 단기 영향 |
| **ARMA(p, q)** | AR + MA | 둘 다 (정상 시계열) |
| **ARIMA(p, d, q)** | $\Delta^d y_t \sim \text{ARMA}(p, q)$ | 비정상 시계열 (추세) |
| **SARIMA(p,d,q)(P,D,Q,m)** | ARIMA + 계절 차분 + 계절 AR/MA | 비정상 + 계절성 |
| **SARIMAX** | SARIMA + 외생변수 | 위 + 보조 입력 변수 |

다음 포스팅에서는 이 모형들을 실제 데이터에 적용하는 흐름 — 데이터 진단부터 모형 선택, 잔차 진단, 예측까지 — 을 처음부터 끝까지 따라가 보겠습니다.

---

[^1]: 백색잡음(white noise)은 평균 0, 분산이 일정한 상수이며, 시점 간 상관이 없는 확률과정입니다. AR/MA 모형의 오차항 가정이며, 이 가정이 깨지면 모형의 통계적 성질이 보장되지 않습니다.

[^2]: PACF의 정확한 정의는 "$y_t$와 $y_{t-k}$ 사이에서 $y_{t-1}, \ldots, y_{t-k+1}$의 선형 영향을 모두 제거한 후의 상관계수"입니다. 계산 방법으로는 Yule-Walker 방정식이나 Durbin 알고리즘을 씁니다. 자세한 수식은 Box, Jenkins, Reinsel & Ljung(2015) Ch. 3 참고.

[^3]: 이 표는 Box-Jenkins 식별 절차의 핵심입니다. 자세한 논의와 예시는 Hamilton(1994) Ch. 4 또는 Box et al.(2015) Ch. 6에서 다룹니다.

[^4]: AIC(Akaike Information Criterion), BIC(Bayesian Information Criterion)는 모형의 가능도와 모수 개수를 함께 고려해 모형을 비교하는 기준입니다. 일반적으로 BIC가 AIC보다 더 단순한 모형을 선택하는 경향이 있습니다.

[^5]: Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.), Section 9.1. 무료 온라인 교재로 공개되어 있습니다 (otexts.com/fpp3).

[^6]: 외생변수(exogenous variable)란 모형 안에서 결정되지 않고 외부에서 주어지는 변수를 의미합니다. 반대 개념은 내생변수(endogenous variable)입니다. SARIMAX의 외생변수는 정확히 말해 "조건부 외생(conditionally exogenous)" 가정 하에 다뤄지며, 자세한 논의는 Hamilton(1994) Ch. 8 참고.

---

**참고문헌**

- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
- Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting* (3rd ed.). Springer.

---

AI의 도움을 받아 작성되었으며 최대한 레퍼런스를 밝히려 노력했으나 오류가 있을 수 있으니 정확한 정보를 다시 한번 확인하시기 바랍니다.
