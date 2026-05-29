---
title: "베이지안 회귀와 MAP: L1·L2 규제는 왜 사전 분포인가"
date: 2026-05-27 10:00:00 +0900
categories: [Statistics, Machine Learning]
tags: [bayesian, MAP, prior, posterior, likelihood, ridge, lasso, gaussian, laplace, regularization]
math: true
---

## 서론

[L1·L2 규제 포스팅]({% post_url 2026-05-27-l1-l2-regularization %})의 마지막에 이런 말을 남겼습니다.

> L2 = 가우시안 사전 분포, L1 = 라플라스 사전 분포. 베이즈 관점에서 규제는 MAP 추정과 동치다.

이 포스팅에서는 그 문장이 왜 성립하는지를 처음부터 차근차근 풀어냅니다.

이번 글의 흐름은 이렇습니다.

1. 베이지안 회귀가 무엇인지 — 빈도주의와 무엇이 다른가
2. MAP 추정이 무엇인지 — 왜 규제와 연결되는가
3. **왜 L2는 가우시안, L1은 라플라스 사전인가** — 수식과 그래프로 확인

---

## 1. 빈도주의 vs 베이지안 — 파라미터를 바라보는 두 시각

### 1.1 빈도주의 (Frequentist)

빈도주의 관점에서 **파라미터 $\mathbf{w}$는 고정된 미지의 상수**입니다.  
데이터는 확률적으로 생성되고, 우리는 그 데이터로부터 $\mathbf{w}$를 최대한 정확하게 추정합니다.

- OLS, MLE가 여기에 해당
- "참값"이 하나 있고, 우리가 그것을 모를 뿐

### 1.2 베이지안 (Bayesian)

베이지안 관점에서 **파라미터 $\mathbf{w}$ 자체가 확률 변수**입니다.  
데이터를 보기 전에도 $\mathbf{w}$에 대한 믿음(**사전 분포**, prior)이 있고,  
데이터를 본 뒤 그 믿음을 업데이트한 것이 **사후 분포**(posterior)입니다.

---

## 2. 사전 분포·사후 분포·베이즈 정리

### 2.1 사전 분포 (Prior)

**사전 분포 $p(\mathbf{w})$** 는 데이터를 보기 *전*, 파라미터 $\mathbf{w}$에 대해 우리가 갖고 있는 믿음을 확률 분포로 표현한 것입니다.

예를 들어 "회귀 계수들은 아마 0 근방에 있을 것이다"라고 생각한다면, 평균 0의 좁은 가우시안을 사전 분포로 쓸 수 있습니다. 반대로 아무 정보가 없다면 값에 상관없이 동일한 확률을 주는 균등 분포(flat prior)를 쓸 수 있습니다.

핵심은 사전 분포가 **데이터와 무관하게** 먼저 지정된다는 점입니다. 분석자의 사전 지식이나 도메인 이해를 모델에 주입하는 수단입니다.

### 2.2 가능도 (Likelihood)

**가능도 $p(\mathbf{y} \mid X, \mathbf{w})$** 는 파라미터 $\mathbf{w}$가 고정되어 있을 때 이 데이터 $\mathbf{y}$가 나올 확률입니다.

사전 분포가 "데이터를 보기 전 믿음"이라면, 가능도는 "데이터가 파라미터에 대해 무엇을 말해주는가"입니다.

### 2.3 사후 분포 (Posterior)

**사후 분포 $p(\mathbf{w} \mid X, \mathbf{y})$** 는 데이터를 본 *후* 업데이트된 믿음입니다.

$$
\underbrace{p(\mathbf{w} \mid X, \mathbf{y})}_{\text{사후(posterior)}}
\;=\;
\frac{
  \overbrace{p(\mathbf{y} \mid X, \mathbf{w})}^{\text{가능도(likelihood)}}
  \;\times\;
  \overbrace{p(\mathbf{w})}^{\text{사전(prior)}}
}{
  \underbrace{p(\mathbf{y} \mid X)}_{\text{정규화 상수}}
}
$$

직관적으로 읽으면 이렇습니다.

> 사후 = (데이터가 $\mathbf{w}$를 얼마나 지지하는가) × (원래 $\mathbf{w}$를 얼마나 믿었는가)

분모 $p(\mathbf{y} \mid X)$는 $\mathbf{w}$와 무관한 정규화 상수이므로, 최적화할 때는 무시해도 됩니다.

$$
p(\mathbf{w} \mid X, \mathbf{y}) \;\propto\; p(\mathbf{y} \mid X, \mathbf{w}) \;\times\; p(\mathbf{w})
$$

데이터가 많아질수록 가능도 항이 지배적이 되어 **사후 분포가 점점 좁아집니다** — 데이터가 쌓일수록 불확실성이 줄고 확신이 강해지는 현상이 수식으로 표현된 것입니다.

### 2.4 MLE와의 결정적 차이

| | 사전 분포 | 사후 분포 | 최대화 대상 |
|---|---|---|---|
| **MLE** | 없음 | 없음 | 가능도 $p(\mathbf{y} \mid X, \mathbf{w})$ |
| **MAP** | **있음** | **있음** (최빈값만 사용) | 사후 분포 $p(\mathbf{w} \mid X, \mathbf{y})$ |
| **완전 베이지안** | 있음 | 있음 (분포 전체 유지) | 해당 없음 (적분) |

MLE는 $\mathbf{w}$를 확률 변수로 보지 않습니다. 사전 분포를 지정할 이유도, 사후 분포를 계산할 이유도 없습니다. 그냥 "이 데이터가 나올 가능성을 가장 크게 만드는 $\mathbf{w}$는 무엇인가"만 묻습니다.

MAP는 사전 분포를 지정하고, 그것과 가능도를 곱해 사후 분포를 구성한 뒤, 그 사후 분포의 **최빈값(mode)** 을 최종 추정값으로 씁니다. 사후 분포를 "계산은 하되, 꼭짓점 하나만 취하고 나머지는 버리는" 방식입니다.

> MLE에 사전 분포를 하나 추가하면 MAP가 됩니다. 그리고 그 사전 분포의 모양이 곧 규제의 종류를 결정합니다.

---

## 3. 선형 회귀의 가능도 — 가우시안 노이즈 가정

선형 회귀 모델은 다음과 같이 씁니다.

$$
y_i = \mathbf{w}^\top \mathbf{x}_i + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0,\, \sigma^2)
$$

잡음 $\varepsilon_i$이 가우시안이면, $y_i$도 조건부 가우시안입니다.

$$
p(y_i \mid \mathbf{x}_i, \mathbf{w}) = \mathcal{N}(y_i;\; \mathbf{w}^\top \mathbf{x}_i,\; \sigma^2)
= \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y_i - \mathbf{w}^\top \mathbf{x}_i)^2}{2\sigma^2}\right)
$$

샘플이 독립이면 전체 가능도는 곱입니다.

$$
p(\mathbf{y} \mid X, \mathbf{w})
= \prod_{i=1}^n p(y_i \mid \mathbf{x}_i, \mathbf{w})
= \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y_i - \mathbf{w}^\top \mathbf{x}_i)^2}{2\sigma^2}\right)
$$

로그 가능도로 변환하면:

$$
\log p(\mathbf{y} \mid X, \mathbf{w})
= -\frac{1}{2\sigma^2} \underbrace{\sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i)^2}_{\text{RSS}} + \text{const}
$$

**로그 가능도를 최대화 = RSS를 최소화 = OLS**  
즉, OLS는 "잡음이 가우시안이다"라는 가정 아래 MLE와 같습니다(가능도와 MLE의 발상은 [이 글]({% post_url 2026-05-02-parameter-estimation %}) 참고).

---

## 4. MAP 추정 — 사전 분포를 더하면 규제가 된다

MLE는 가능도만 최대화합니다.  
**MAP(Maximum A Posteriori)**는 사후 분포를 최대화합니다.

$$
\hat{\mathbf{w}}_{\text{MAP}}
= \arg\max_{\mathbf{w}}\; p(\mathbf{w} \mid X, \mathbf{y})
= \arg\max_{\mathbf{w}}\; \bigl[\, p(\mathbf{y} \mid X, \mathbf{w}) \cdot p(\mathbf{w}) \,\bigr]
$$

로그를 취해도 최댓값 위치는 바뀌지 않습니다.

$$
\hat{\mathbf{w}}_{\text{MAP}}
= \arg\max_{\mathbf{w}}\;\bigl[\log p(\mathbf{y} \mid X, \mathbf{w}) + \log p(\mathbf{w})\bigr]
$$

3절의 로그 가능도를 대입하면:

$$
= \arg\max_{\mathbf{w}}\;\left[ -\frac{\text{RSS}}{2\sigma^2} + \log p(\mathbf{w}) \right]
$$

최대화를 최소화로 바꾸면:

$$
\hat{\mathbf{w}}_{\text{MAP}}
= \arg\min_{\mathbf{w}}\;\left[ \text{RSS} \;-\; 2\sigma^2 \log p(\mathbf{w}) \right]
$$

> $-2\sigma^2 \log p(\mathbf{w})$ 항이 바로 **규제 패널티**입니다.  
> 사전 분포 $p(\mathbf{w})$의 형태에 따라 패널티의 모양이 결정됩니다.

---

## 5. L2 규제 = 가우시안 사전 분포

### 5.1 가우시안 사전 분포 가정

계수들이 서로 독립적으로 평균 0, 분산 $\tau^2$인 정규분포를 따른다고 가정합니다.

$$
w_j \;\overset{\text{iid}}{\sim}\; \mathcal{N}(0,\, \tau^2)
\qquad\Rightarrow\qquad
p(\mathbf{w}) = \prod_{j=1}^p \frac{1}{\sqrt{2\pi\tau^2}} \exp\!\left(-\frac{w_j^2}{2\tau^2}\right)
$$

### 5.2 로그 사전 분포

$$
\log p(\mathbf{w}) = -\frac{1}{2\tau^2} \sum_{j=1}^p w_j^2 + \text{const}
$$

### 5.3 MAP 목적함수에 대입

$$
\text{RSS} - 2\sigma^2 \log p(\mathbf{w})
= \text{RSS} + \frac{\sigma^2}{\tau^2} \sum_{j=1}^p w_j^2 + \text{const}
$$

$\lambda = \dfrac{\sigma^2}{\tau^2}$으로 놓으면:

$$
\boxed{
\hat{\mathbf{w}}_{\text{MAP}}
= \arg\min_{\mathbf{w}} \left[ \text{RSS} + \lambda \sum_{j=1}^p w_j^2 \right]
= \hat{\mathbf{w}}_{\text{Ridge}}
}
$$

**가우시안 사전 분포를 가정한 MAP 추정 = Ridge 회귀**

### 5.4 왜 가우시안 사전의 -log는 포물선인가

가우시안 pdf의 핵심부는 $\exp(-w^2 / 2\tau^2)$입니다.  
로그를 취하면 $-w^2 / 2\tau^2$ — **$w^2$에 비례하는 이차 함수(포물선)**가 됩니다.

가우시안은 0에서 부드럽게 정점을 찍고 좌우 대칭으로 떨어집니다.  
"작은 $w$는 허용하지만, 클수록 급격히 불가능해진다"는 연속적 압력을 줍니다.  
이것이 Ridge가 계수를 0 가까이 **수축**시키되 **정확히 0으로 만들지 못하는** 이유입니다.

---

## 6. L1 규제 = 라플라스 사전 분포

### 6.1 라플라스 사전 분포 가정

계수들이 서로 독립적으로 위치 모수 0, 척도 모수 $b$인 라플라스 분포를 따른다고 가정합니다.

$$
w_j \;\overset{\text{iid}}{\sim}\; \text{Laplace}(0,\, b)
\qquad\Rightarrow\qquad
p(\mathbf{w}) = \prod_{j=1}^p \frac{1}{2b} \exp\!\left(-\frac{|w_j|}{b}\right)
$$

라플라스 분포의 pdf를 가우시안과 비교하면 두 가지 차이가 있습니다.

- **$w=0$에서 뾰족한 꼭짓점** — 가우시안은 $w=0$에서 미분 가능하지만, 라플라스는 꺾임
- **두꺼운 꼬리(heavy tail)** — 가우시안보다 극단값에 더 많은 확률을 배정

### 6.2 로그 사전 분포

$$
\log p(\mathbf{w}) = -\frac{1}{b} \sum_{j=1}^p |w_j| + \text{const}
$$

### 6.3 MAP 목적함수에 대입

$$
\text{RSS} - 2\sigma^2 \log p(\mathbf{w})
= \text{RSS} + \frac{2\sigma^2}{b} \sum_{j=1}^p |w_j| + \text{const}
$$

$\lambda = \dfrac{2\sigma^2}{b}$으로 놓으면:

$$
\boxed{
\hat{\mathbf{w}}_{\text{MAP}}
= \arg\min_{\mathbf{w}} \left[ \text{RSS} + \lambda \sum_{j=1}^p |w_j| \right]
= \hat{\mathbf{w}}_{\text{Lasso}}
}
$$

**라플라스 사전 분포를 가정한 MAP 추정 = Lasso 회귀**

### 6.4 왜 라플라스 사전의 -log는 V자인가

라플라스 pdf의 핵심부는 $\exp(-|w|/b)$입니다.  
로그를 취하면 $-|w|/b$ — **$|w|$에 비례하는 절댓값 함수(V자)**가 됩니다.

V자 패널티는 $w=0$에서 **미분이 불연속**입니다.  
$w$가 0에서 조금만 움직여도 일정한 기울기 $\pm 1/b$로 패널티가 증가합니다.  
이 날카로운 꺾임 때문에 최적화가 $w=0$에 "붙어버리는" 현상이 발생하고,  
이것이 Lasso가 계수를 **정확히 0으로** 만드는 근본 이유입니다.

---

## 7. 두 사전 분포를 한눈에 — 그림으로 비교

![가우시안 vs 라플라스 사전 분포](/assets/img/posts/fig_bayesian_prior.png)
_왼쪽: 분포 모양. 오른쪽: $-\log p(w)$ = MAP 패널티 모양. 라플라스의 뾰족한 꼭짓점이 V자 패널티로 이어진다._

| | **가우시안 사전** | **라플라스 사전** |
|---|---|---|
| 분포 정점 | 부드러운 곡면 (미분 가능) | **뾰족한 꼭짓점** (미분 불가) |
| $-\log p(w)$ | 포물선 $\propto w^2$ | V자 $\propto \lvert w \rvert$ |
| MAP 패널티 | L2 (Ridge) | L1 (Lasso) |
| 계수 거동 | 0 근방으로 수축 | 일부가 정확히 0 |
| 꼬리 두께 | 얇음 (exponential decay) | 두꺼움 (heavy tail) |

---

## 8. $\lambda$의 베이지안 해석

빈도주의에서 $\lambda$는 "규제 강도"를 조절하는 하이퍼파라미터입니다.  
베이지안 관점에서는 더 구체적인 의미가 있습니다.

**Ridge**의 경우:

$$
\lambda = \frac{\sigma^2}{\tau^2}
$$

- $\sigma^2$: 잡음의 분산 — 데이터가 얼마나 흩어져 있는가
- $\tau^2$: 사전 분포의 분산 — 계수가 얼마나 넓게 퍼져 있을 것이라 믿는가

| 상황 | 의미 | $\lambda$ |
|---|---|---|
| $\tau^2$이 크다 | 계수가 클 수도 있다는 약한 사전 믿음 | 작아짐 → 규제 약해짐 |
| $\tau^2$이 작다 | 계수가 0에 가까울 것이라는 강한 사전 믿음 | 커짐 → 규제 강해짐 |
| $\sigma^2$이 크다 | 데이터가 잡음이 많아 신뢰도 낮음 | 커짐 → 사전 믿음에 의존 |
| $\sigma^2$이 작다 | 데이터가 깨끗하고 신뢰도 높음 | 작아짐 → 데이터를 따름 |

$\lambda$는 "데이터를 얼마나 믿는가"와 "사전 믿음을 얼마나 믿는가"의 비율입니다.

---

## 9. MAP vs 완전 베이지안

MAP는 사후 분포의 **최빈값(mode)** 하나만 취합니다.  
**완전 베이지안(full Bayesian)**은 사후 분포 전체를 계산합니다.

$$
\text{MAP: } \hat{\mathbf{w}} = \arg\max_{\mathbf{w}}\; p(\mathbf{w} \mid X, \mathbf{y})
$$

$$
\text{완전 베이지안: } p(\mathbf{w} \mid X, \mathbf{y}) \text{ 전체를 유지}
$$

완전 베이지안의 예측은 **사후 예측 분포(posterior predictive)**입니다.

$$
p(y^* \mid \mathbf{x}^*, X, \mathbf{y})
= \int p(y^* \mid \mathbf{x}^*, \mathbf{w})\; p(\mathbf{w} \mid X, \mathbf{y})\; d\mathbf{w}
$$

모든 가능한 $\mathbf{w}$ 값을 평균 내는 셈입니다.  
가우시안 가능도 + 가우시안 사전 분포 조합(켤레 사전, conjugate prior)에서는 이 적분이 해석적으로 풀립니다.

| | MAP | 완전 베이지안 |
|---|---|---|
| 출력 | 점 추정 (최빈값) | 분포 전체 |
| 불확실성 | 없음 | 있음 (신뢰 구간 포함) |
| 계산 비용 | 낮음 | 높음 (적분 또는 MCMC) |
| Ridge/Lasso와 동치 | **예** | 아니오 |

Ridge와 Lasso는 MAP까지만 대응됩니다.  
"계수의 불확실성이 얼마나 되는가"를 알려면 완전 베이지안이 필요합니다.

---

## 마치며

정리하면:

1. **가우시안 잡음을 가정한 선형 회귀에서 MLE = OLS**
2. **여기에 가우시안 사전 분포를 추가해 MAP를 구하면 = Ridge**
3. **라플라스 사전 분포를 추가해 MAP를 구하면 = Lasso**

규제는 단순한 과적합 방지 트릭이 아닙니다.  
"계수가 어떻게 분포할 것이다"라는 **사전 믿음을 모델에 명시적으로 주입하는 행위**입니다.

가우시안 사전의 부드러운 정점 → L2 포물선 패널티 → 계수 수축  
라플라스 사전의 뾰족한 꼭짓점 → L1 V자 패널티 → 계수 소거

분포의 **모양**이 곧 규제의 **성질**을 결정합니다.
