---
title: "지수평활부터 ETS까지 — 단변량 시계열의 또 다른 표준"
date: 2026-05-08 09:00:00 +0900
categories: [Time Series]
tags: [exponential-smoothing, ets, holt-winters, smoothing, time-series]
math: true
---

지난 시리즈에서 ARIMA/SARIMA 한 묶음을 끝까지 다뤘습니다. 그런데 시계열 분야에는 ARIMA와 거의 동급으로 표준 취급받는 모형군이 하나 더 있습니다. **지수평활(Exponential Smoothing)** 과 그 정점인 **ETS** 입니다. Hyndman & Athanasopoulos(2021)의 *Forecasting: Principles and Practice* 에서 ARIMA보다 먼저 다루는 챕터이기도 합니다.

이 글에서는 가장 단순한 단순지수평활(SES)에서 시작해서, 추세와 계절성을 차례로 더해 가며 ETS까지 도달하는 단계적 빌드업을 따라가 보겠습니다. 마지막에는 같은 데이터(Air Passengers)에서 ETS와 SARIMA를 직접 비교해 보겠습니다.

## 1. 지수평활의 발상 — "최근 값을 더 무겁게"

직관부터 시작합니다. 지금까지의 시계열 $$y_1, y_2, \ldots, y_t$$를 가지고 다음 값 $$\hat{y}_{t+1}$$을 예측한다고 합시다.

가장 단순한 두 가지 발상은:

- **(전체 평균)** $$\hat{y}_{t+1} = \bar{y} = \frac{1}{t} \sum_{i=1}^{t} y_i$$
- **(직전 값)** $$\hat{y}_{t+1} = y_t$$

전체 평균은 모든 과거를 똑같이 취급해 너무 둔감하고, 직전 값만 쓰는 건 노이즈에 너무 민감합니다. 그 사이의 절충이 **이동평균**입니다.

$$
\hat{y}_{t+1} = \frac{1}{k} \sum_{i=t-k+1}^{t} y_i \quad (\text{윈도우 크기 } k)
$$

이건 합리적이지만 한 가지 어색한 점이 있습니다 — **윈도우 안의 값들을 모두 동등하게 취급**한다는 점입니다. 직관적으로는 어제 값이 일주일 전 값보다 더 중요할 텐데요.

지수평활은 이 직관을 가장 깔끔하게 식으로 옮깁니다. **최근 값에 더 큰 가중치, 과거로 갈수록 기하급수적으로 감쇠하는 가중치**를 부여합니다.

$$
\hat{y}_{t+1} = \alpha y_t + \alpha(1-\alpha) y_{t-1} + \alpha(1-\alpha)^2 y_{t-2} + \cdots
$$

가중치 비율 $$\alpha, \alpha(1-\alpha), \alpha(1-\alpha)^2, \ldots$$는 1, $$(1-\alpha)$$, $$(1-\alpha)^2, \ldots$$의 등비수열이고, 모두 합하면 1입니다. **수학적으로 깔끔한 가중평균**입니다.

이동평균과 지수평활을 같은 데이터에 적용해 보면:

![이동평균 vs 지수평활](/assets/img/posts/fig_smoothing_ma_vs_ses.png)

두 곡선이 비슷해 보이지만 미묘한 차이가 있습니다. 이동평균은 윈도우의 양 끝에서 한 점이 들어오거나 빠지면서 약간의 출렁임이 있고, 지수평활은 더 부드럽게 이어집니다. 또 이동평균은 윈도우 크기만큼 시작 부분에서 결측이 생기는 반면, 지수평활은 첫 시점부터 값을 줍니다.

## 2. SES — 단순지수평활(Simple Exponential Smoothing)

위의 무한합을 매번 계산할 필요는 없습니다. **재귀 형태**로 간단히 표현됩니다.

$$
\hat{y}_{t+1} = \alpha y_t + (1-\alpha) \hat{y}_t
$$

이 한 줄이 SES의 전부입니다. 새 예측값은 **(이번 관측값과 직전 예측값의 가중평균)**. $$\alpha$$는 평활 모수(smoothing parameter)이고 $$0 < \alpha < 1$$ 범위입니다.[^1]

$$\alpha$$ 값에 따라 평활 정도가 어떻게 달라지는지 보겠습니다.

![SES의 알파 효과](/assets/img/posts/fig_ses_alpha.png)

- **$$\alpha = 0.1$$ (위)**: 새 관측에 거의 반응 안 함. 매우 부드럽지만 변화에 늦게 따라감.
- **$$\alpha = 0.5$$ (중간)**: 균형 잡힘.
- **$$\alpha = 0.9$$ (아래)**: 거의 노이즈를 그대로 따라감. 평활 효과가 거의 없음.

> **함정 박스 1 — α는 어떻게 정하는가**
> 
> 실무에서 $$\alpha$$는 보통 **데이터로부터 추정**합니다. 학습 구간의 잔차 제곱합을 최소화하는 $$\alpha$$를 찾는 식입니다. statsmodels의 `SimpleExpSmoothing(...).fit()`이 자동으로 이 일을 해 줍니다. 수동 지정도 가능하지만, 추정 결과가 보통 더 좋습니다.

statsmodels로 적용하면 한 줄입니다.

```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

model = SimpleExpSmoothing(series).fit()  # α 자동 추정
forecast = model.forecast(steps=12)
```

여기서 한계가 보입니다. **SES의 예측은 한 시점 이후로는 모두 같은 값(평탄선)** 입니다. $$\hat{y}_{t+1} = \hat{y}_{t+2} = \hat{y}_{t+3} = \cdots$$. SES는 본질적으로 "수준(level)"만 추적하기 때문에 추세를 모릅니다. 그래서 다음 단계가 필요합니다.

## 3. Holt's Linear Trend — 추세 추가

추세가 있는 시계열을 다루려면 **수준 외에 기울기도 추적**해야 합니다. Holt(1957)가 제안한 방식이 SES에 추세 컴포넌트를 더하는 것입니다.[^2]

$$
\begin{aligned}
\ell_t &= \alpha y_t + (1-\alpha)(\ell_{t-1} + b_{t-1}) \\
b_t &= \beta(\ell_t - \ell_{t-1}) + (1-\beta) b_{t-1} \\
\hat{y}_{t+h} &= \ell_t + h \cdot b_t
\end{aligned}
$$

세 식의 의미는:

- $$\ell_t$$: 시점 $$t$$의 **수준(level)** — "현재 가치가 얼마인가"
- $$b_t$$: 시점 $$t$$의 **추세(trend)** — "단위 시간당 얼마나 변하는가"
- 예측: $$h$$ 시점 후의 값은 현재 수준 + $$h$$만큼의 추세

평활 모수가 두 개($$\alpha, \beta$$)로 늘었습니다. $$\alpha$$는 수준의 평활 정도, $$\beta$$는 추세의 평활 정도입니다. 둘 다 데이터로부터 추정합니다.

이렇게 하면 SES와 달리 예측값이 평탄하지 않고 **선형으로 외삽**됩니다.

## 4. Holt-Winters — 계절성 추가

계절성까지 다루려면 컴포넌트를 하나 더 추가합니다. Holt(1957)와 Winters(1960)의 결합이라 **Holt-Winters** 또는 **삼중지수평활(triple exponential smoothing)** 로 불립니다.[^3]

가법 계절성 버전:

$$
\begin{aligned}
\ell_t &= \alpha (y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1}) \\
b_t &= \beta(\ell_t - \ell_{t-1}) + (1-\beta) b_{t-1} \\
s_t &= \gamma (y_t - \ell_t) + (1-\gamma) s_{t-m} \\
\hat{y}_{t+h} &= \ell_t + h \cdot b_t + s_{t+h-m}
\end{aligned}
$$

세 평활 모수($$\alpha, \beta, \gamma$$)와 한 개의 구조 모수($$m$$, 계절 주기)가 들어갑니다. **수준, 추세, 계절성 세 컴포넌트가 각자 자기 평활을 따로 합니다.**

곱셈 계절성 버전은 식의 형태만 살짝 다릅니다 (덧셈을 곱셈/나눗셈으로 대체). Air Passengers처럼 계절 변동의 폭이 수준에 비례해 커지는 경우는 곱셈을 씁니다.

statsmodels로 적용하면 이렇게 됩니다.

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(
    train,
    trend='add',          # 'add' | 'mul' | None
    seasonal='mul',       # 'add' | 'mul' | None
    seasonal_periods=12,
).fit()
forecast = model.forecast(steps=24)
```

이전 글에서 SARIMA로 다뤘던 Air Passengers에 Holt-Winters(가법 추세 + 곱셈 계절성)를 적용해 보면:

![Holt-Winters on Air Passengers](/assets/img/posts/fig_hw_airpass.png)

24개월 예측 MAPE가 **6.39%** 입니다. 이전 글에서 SARIMA(0,1,1)(0,1,1)$$_{12}$$이 같은 데이터에서 얻은 8.52%보다 살짝 좋습니다. 모형의 가정과 데이터의 성격이 잘 맞으면 Holt-Winters가 SARIMA를 이기는 경우도 흔합니다.

> **함정 박스 2 — 가법 vs 곱셈 컴포넌트 선택**
> 
> 추세는 보통 가법으로 충분합니다. 헷갈리는 부분은 계절성입니다. **계절 변동의 폭이 시간에 따라 일정하면 가법, 수준에 비례해 커지면 곱셈**입니다. Air Passengers는 명확히 곱셈입니다(이전 SARIMA 글에서도 같은 판정). 헷갈리면 두 버전을 다 적합해 보고 잔차의 패턴이 더 깨끗한 쪽을 고르세요.

## 5. ETS — 상태공간 정식화

여기까지 SES → Holt → Holt-Winters로 컴포넌트를 하나씩 더해 왔습니다. **ETS (Error-Trend-Seasonal)** 는 이 모든 변형을 하나의 통일된 틀에 담은 것입니다. Hyndman 등이 2002년에 정식화했고[^4], 현재 시계열 예측의 양대 표준 중 하나입니다.

ETS의 표기는 세 글자입니다.

$$
\text{ETS}(\underbrace{\cdot}_{\text{Error}}, \underbrace{\cdot}_{\text{Trend}}, \underbrace{\cdot}_{\text{Seasonal}})
$$

각 자리에 들어갈 수 있는 값:

| 컴포넌트 | 가능한 값 | 의미 |
|---|---|---|
| **Error** | A, M | 가법(additive) 또는 곱셈(multiplicative) |
| **Trend** | N, A, A$$_d$$ | 없음 / 가법 / 가법 감쇠(damped) |
| **Seasonal** | N, A, M | 없음 / 가법 / 곱셈 |

조합하면 **30개 모델군**이 됩니다.[^5] 익숙한 이름들과의 대응:

| ETS 표기 | 별명 |
|---|---|
| ETS(A, N, N) | SES (단순지수평활) |
| ETS(A, A, N) | Holt's linear trend |
| ETS(A, A, A) | Holt-Winters 가법 |
| ETS(A, A, M) | Holt-Winters 곱셈 |
| ETS(A, A$$_d$$, A) | 감쇠 추세 + 가법 계절성 |

ETS의 진짜 강점은 **상태공간 모형(state space model)** 으로 정식화됐다는 점입니다.[^6] 이게 두 가지 실용적 이점을 줍니다.

1. **신뢰구간을 통계적으로 정확히 계산**할 수 있음 — 기존 Holt-Winters는 점예측만 줄 수 있었지만, ETS는 가능도 기반 신뢰구간을 줍니다.
2. **AIC/BIC로 30개 모델 중 자동 선택** 가능 — 어떤 컴포넌트 조합이 데이터에 맞는지 정보 기준으로 비교할 수 있습니다.

```python
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# 수동 지정
model = ETSModel(train, error='add', trend='add', seasonal='mul', 
                 seasonal_periods=12).fit()

# 또는 R의 forecast::ets()와 statsforecast 패키지의
# AutoETS는 30개 모델을 자동 비교해 줍니다.
```

> **함정 박스 3 — 30개 중 어떤 걸 쓰나**
> 
> 자동 선택(R의 `forecast::ets()`나 Python의 `statsforecast.AutoETS`)이 보통 합리적인 답을 줍니다. 다만 자동 선택이 곱셈 계절성을 고른다면 데이터가 양수여야 합니다 — 음수가 섞인 시계열에서는 가법으로 강제하셔야 합니다. 또 표본 크기가 너무 작으면(2주기 미만) 계절 컴포넌트를 신뢰성 있게 추정할 수 없으니 주기를 잡지 마시기 바랍니다.

## 6. ETS vs SARIMA — 같은 데이터, 두 가지 접근

이제 공정한 비교를 해 보겠습니다. 이전 글에서 SARIMA(0,1,1)(0,1,1)$$_{12}$$로 Air Passengers를 24개월 예측했을 때 MAPE 8.52%를 얻었습니다. 같은 train/test split에 ETS(A,A,M)를 적용해 봅니다.

![ETS vs SARIMA](/assets/img/posts/fig_ets_vs_sarima.png)

| 모델 | 24개월 예측 MAPE |
|---|---:|
| Holt-Winters (수동) | 6.39% |
| **ETS(A, A, M)** | **6.56%** |
| SARIMA(0,1,1)(0,1,1)$$_{12}$$ | 8.52% |

**이 데이터에서는 ETS가 SARIMA보다 살짝 좋습니다.** 그렇다고 ETS가 항상 SARIMA보다 좋다는 뜻은 아닙니다. 두 모형은 서로 잘하는 영역이 다릅니다.

| | ARIMA / SARIMA | ETS |
|---|---|---|
| 사고 방식 | "차분 후 정상 시계열의 자기상관" | "수준·추세·계절성을 직접 평활" |
| 잘 다루는 데이터 | 자기상관 구조가 분명한 시계열 | 추세와 계절성이 명확한 시계열 |
| 정상성 가정 | 차분으로 정상화 필요 | 비정상 시계열도 자연스럽게 |
| 모수 의미 | $$\phi, \theta$$ (해석 어려움) | $$\alpha, \beta, \gamma$$ (직관적) |
| 자동 선택 | `auto_arima` (pmdarima) | `AutoETS` (statsforecast) |
| 여러 시계열에 일괄 적용 | 무거움 | 가벼움 |

**실무 권고**:

- 어떤 게 좋을지 모르겠으면 **둘 다 적합해 보고 검증 셋의 성능으로 비교**하세요. M3, M4 같은 forecasting 대회에서도 단일 모형이 항상 이기는 일은 없고, **두 모형의 단순 평균이 종종 최고 성능**을 보입니다.[^7]
- 자기상관 구조가 복잡한 시계열(거시경제 변수, 금융)은 ARIMA/SARIMA가 강합니다.
- 추세·계절성이 명확하고 자기상관은 단순한 시계열(소매 매출, 트래픽)은 ETS가 강합니다.
- 여러 시계열을 일괄 처리해야 한다면 ETS가 보통 더 가볍고 빠릅니다.

## 7. 정리 — 무엇을 배웠는가

| 모형 | 추가된 컴포넌트 | 평활 모수 |
|---|---|---|
| SES (지수평활) | 수준 | $$\alpha$$ |
| Holt | 수준 + 추세 | $$\alpha, \beta$$ |
| Holt-Winters | 수준 + 추세 + 계절성 | $$\alpha, \beta, \gamma$$ |
| ETS | 위 모두를 30가지로 일반화 + 상태공간 | 동일 + 자동 선택 |

세 줄로 요약하면:

1. 지수평활은 **최근 값에 더 큰 가중치**를 주는 단순한 발상에서 출발해 컴포넌트를 더해 가며 ETS까지 도달합니다.
2. ETS는 ARIMA와 함께 단변량 시계열의 양대 표준입니다. **데이터에 따라 어느 쪽이 나을지가 다릅니다.**
3. statsforecast의 `AutoETS`나 R의 `forecast::ets()`는 30개 모델 중 적절한 것을 자동으로 골라 줍니다.

다음 글에서는 이 미니 시리즈의 2편으로, **Prophet의 내부 구조**를 들여다보겠습니다. Prophet은 사실 새로운 마법이 아니라 (1) 가법 분해, (2) 푸리에 급수로 계절성, (3) piecewise linear trend, (4) 베이지안 추정 — 이 네 가지의 영리한 조합입니다. 신호처리에 익숙하신 분이면 푸리에 부분에서 친숙한 광경을 보시게 될 것입니다.

---

[^1]: SES는 ARIMA(0,1,1)과 수학적으로 등가입니다. 즉 차분된 시계열이 MA(1)을 따른다고 가정하는 모형의 평활 형태입니다. 자세한 도출은 Hyndman et al.(2008) Ch. 11 참고.

[^2]: Holt, C. C. (1957). "Forecasting seasonals and trends by exponentially weighted moving averages." 처음에는 ONR 메모로 발표됐다가 2004년 *International Journal of Forecasting*에 재출간되었습니다.

[^3]: Winters, P. R. (1960). "Forecasting sales by exponentially weighted moving averages." *Management Science*, 6(3), 324–342. Holt가 추세를, Winters가 계절성을 각각 추가한 형태로 결합되어 "Holt-Winters" 모형으로 불리게 되었습니다.

[^4]: Hyndman, R. J., Koehler, A. B., Snyder, R. D., & Grose, S. (2002). "A state space framework for automatic forecasting using exponential smoothing methods." *International Journal of Forecasting*, 18(3), 439–454.

[^5]: 2 (error) × 3 (trend) × 3 (seasonal) = 18가지가 기본이고, error와 seasonal의 호환성 조건(곱셈 계절성에 가법 오차 등) 때문에 실제 유효한 모델은 더 적습니다. 자세한 분류는 Hyndman & Athanasopoulos(2021) Section 8.6 참고.

[^6]: 상태공간 모형(state space model)은 관측 가능한 변수($$y_t$$)와 관측 불가능한 상태 변수들($$\ell_t, b_t, s_t$$)을 분리하여, 상태가 시간에 따라 어떻게 진화하고 그것이 관측치로 어떻게 드러나는지를 두 종류의 방정식(state equation, observation equation)으로 표현하는 틀입니다. 칼만 필터가 같은 틀의 대표적 추정 도구입니다.

[^7]: Makridakis, S., & Hibon, M. (2000). "The M3-Competition: results, conclusions and implications." *International Journal of Forecasting*, 16(4), 451–476. 이후 M4 (2018), M5 (2020) 대회에서도 단순 모형의 조합이 복잡한 단일 모형 못지않게 좋은 성능을 보인다는 결과가 일관되게 확인되었습니다.

---

**참고문헌**

- Holt, C. C. (1957). Forecasting seasonals and trends by exponentially weighted moving averages. (ONR Research Memorandum, Carnegie Institute of Technology, 52). Reprinted in *International Journal of Forecasting*, 20(1), 5–10 (2004).
- Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
- Makridakis, S., & Hibon, M. (2000). The M3-Competition: results, conclusions and implications. *International Journal of Forecasting*, 16(4), 451–476.
- Winters, P. R. (1960). Forecasting sales by exponentially weighted moving averages. *Management Science*, 6(3), 324–342.

---

AI의 도움을 받아 작성되었으며 최대한 레퍼런스를 밝히려 노력했으나 오류가 있을 수 있으니 정확한 정보를 다시 한번 확인하시기 바랍니다.
