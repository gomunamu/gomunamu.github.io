---
title: "베이지안 추론 1 — 믿음을 갱신하는 기계"
date: 2026-05-29 09:00:00 +0900
categories: [Machine Learning]
tags: [bayesian, prior, posterior, likelihood, beta-binomial, conjugate, updating]
math: true
published: false
---

## 핵심 아이디어

베이지안 추론은 한 문장으로 요약됩니다.

> **데이터를 보기 전 믿음 × 데이터가 말하는 것 → 데이터를 본 후 믿음**

수식으로:

$$
\underbrace{p(\theta \mid \mathcal{D})}_{\text{posterior}} \;\propto\; \underbrace{p(\mathcal{D} \mid \theta)}_{\text{likelihood}} \;\times\; \underbrace{p(\theta)}_{\text{prior}}
$$

- **Prior** $p(\theta)$: 데이터 없이 $\theta$에 대해 갖고 있는 믿음
- **Likelihood** $p(\mathcal{D} \mid \theta)$: $\theta$가 주어졌을 때 이 데이터가 나올 가능성
- **Posterior** $p(\theta \mid \mathcal{D})$: 데이터를 본 후 갱신된 믿음

---

## 빈도주의 vs 베이지안 — 파라미터를 보는 눈

| | 빈도주의 | 베이지안 |
|---|---|---|
| $\theta$의 성격 | 고정된 미지의 상수 | 확률 변수 |
| 불확실성의 위치 | 데이터 (반복 실험) | 파라미터 자체 |
| 추론 결과 | 점추정 + 신뢰구간 | 분포 (posterior) |

빈도주의는 보통 "참값이 하나 있는데 우리가 모를 뿐"이라는 관점에서 출발합니다.  
베이지안은 파라미터에 대한 불확실성을 분포로 명시적으로 표현합니다.

---

## 예시: 동전 던지기

동전의 앞면 확률 $\theta$를 추정한다고 합시다.

### Prior

아무것도 모를 때: $\theta \sim \text{Beta}(1, 1)$ — 균등분포, 0~1 사이 모두 동등하게 가능.

$$
p(\theta) = 1, \quad \theta \in [0, 1]
$$

### Likelihood

동전을 10번 던져 7번 앞면이 나왔다면:

$$
p(\mathcal{D} \mid \theta) = \binom{10}{7} \theta^7 (1-\theta)^3
$$

### Posterior

Beta 분포와 이항 분포는 **conjugate** 관계 — posterior도 Beta:

$$
p(\theta \mid \mathcal{D}) = \text{Beta}(1+7,\; 1+3) = \text{Beta}(8, 4)
$$

| | 값 |
|---|---|
| Prior 평균 | 0.5 |
| Posterior 평균 | $8/(8+4) \approx 0.667$ |
| MLE | $7/10 = 0.7$ |

Prior가 데이터와 섞여 posterior를 중간 어딘가로 당깁니다.

---

## 데이터가 쌓이면 Prior는 흐려진다

$n$번 던져 $k$번 앞면: posterior는 $\text{Beta}(1+k,\; 1+n-k)$

데이터가 많아질수록 posterior의 분산이 줄어들고, prior의 영향은 대개 희석됩니다.

정규성 조건과 모델이 잘 지정되었다는 가정 아래에서는 posterior가 점점 한 점 근처로 집중하는 현상이 나타납니다.

$$
n \to \infty \implies p(\theta \mid \mathcal{D}) \text{ concentrates near } \theta_0
$$

실무적으로는 **데이터가 충분할수록 prior보다 likelihood가 지배적이 되어, 베이지안 추정과 빈도주의 추정이 비슷해지는 경우가 많습니다.**

---

## 학습 = 분포를 좁혀가는 과정

베이지안 관점에서 "학습"이란 prior → posterior로 분포가 좁아지는 과정입니다.

- 처음엔 넓은 prior (불확실)
- 데이터를 볼 때마다 posterior가 갱신
- 오늘의 posterior가 내일의 prior

동일한 likelihood factorization이 성립한다면, 이 갱신 루프는 순차적으로 적용해도 한꺼번에 적용해도 같은 posterior를 얻습니다.

---

## 다음

[2편]({% post_url 2026-05-29-bayesian-inference-2-prior %})에서는 prior를 어떻게 설계하는가 — 사전지식을 어떻게 수식으로 표현하는가를 다룹니다.
