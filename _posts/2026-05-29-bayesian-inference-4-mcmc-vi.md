---
title: "베이지안 추론 4 — MCMC와 Variational Inference"
date: 2026-05-29 12:00:00 +0900
categories: [Machine Learning]
tags: [mcmc, variational-inference, elbo, kl-divergence, hmc, nuts, pymc, posterior, approximate-inference]
math: true
published: false
---

## 문제: Posterior를 어떻게 구하나

베이즈 정리는 간단합니다.

$$
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\, p(\theta)}{p(\mathcal{D})}
$$

문제는 분모 $p(\mathcal{D})$입니다.

$$
p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta)\, p(\theta)\, d\theta
$$

이 적분은 대부분의 실제 모델에서 **닫힌 형태로 계산하기 어렵습니다** (intractable).  
Conjugate prior가 있는 일부 경우에는 해석적으로 풀 수 있습니다.

---

## 접근 1: MCMC (Markov Chain Monte Carlo)

### 아이디어

$p(\theta \mid \mathcal{D})$를 직접 계산하지 않고, **그 분포에서 샘플링**합니다.  
샘플이 충분하면 분포의 모양을 추정할 수 있습니다.

### Metropolis-Hastings 직관

1. 현재 위치 $\theta_t$에서 후보 $\theta^*$를 제안
2. $\frac{p(\theta^* \mid \mathcal{D})}{p(\theta_t \mid \mathcal{D})}$ 비율을 계산
3. 비율이 1보다 크면 무조건 이동, 작으면 확률적으로 이동
4. 오래 걷다 보면 걸어간 경로의 분포 ≈ posterior

$p(\mathcal{D})$는 비율을 구할 때 **약분되어 사라집니다** — 이것이 핵심.

### HMC / NUTS

Metropolis-Hastings의 문제: 고차원에서 랜덤 워크가 매우 비효율적.

**HMC (Hamiltonian Monte Carlo)**: 물리학의 해밀턴 역학을 이용해 gradient 방향으로 효율적으로 이동.

$$
\text{gradient of } \log p(\theta \mid \mathcal{D}) \text{ 를 따라 이동}
$$

**NUTS (No-U-Turn Sampler)**: HMC의 경로 길이를 적응적으로 정해 튜닝 부담을 줄여줍니다. PyMC, Stan에서 널리 기본 선택지로 쓰입니다.

### MCMC의 특징

| 장점 | 단점 |
|---|---|
| 충분히 수렴하면 posterior expectation을 일관되게 근사 | 느림 (특히 대규모 데이터) |
| 복잡한 모델에도 적용 가능 | Convergence 진단 필요 |
| 불확실성 정량화 정확 | 고차원에서 mixing 문제 |

---

## 접근 2: Variational Inference (VI)

### 아이디어

Posterior $p(\theta \mid \mathcal{D})$를 직접 구하지 않고,  
다루기 쉬운 분포족 $q(\theta; \phi)$ 중에서 **posterior와 가장 비슷한 것**을 찾습니다.

$$
q^*(\theta) = \arg\min_{q \in \mathcal{Q}} D_{\text{KL}}(q(\theta) \| p(\theta \mid \mathcal{D}))
$$

이것은 **최적화 문제**입니다 — 샘플링이 아니라 경사하강법.

### ELBO

KL divergence를 직접 최소화하면 또 $p(\mathcal{D})$가 필요합니다.  
대신 **ELBO (Evidence Lower BOund)** 를 최대화합니다.

$$
\mathcal{L}(\phi) = \mathbb{E}_{q}[\log p(\mathcal{D}, \theta)] - \mathbb{E}_{q}[\log q(\theta)]
$$

$$
= \underbrace{\mathbb{E}_{q}[\log p(\mathcal{D} \mid \theta)]}_{\text{데이터 설명력}} - \underbrace{D_{\text{KL}}(q(\theta) \| p(\theta))}_{\text{prior로부터 멀어진 정도}}
$$

- 첫 번째 항: q로 데이터를 잘 설명해야 함
- 두 번째 항: q가 prior에서 너무 벗어나지 않아야 함

ELBO 최대화 = KL 최소화 (분모 $p(\mathcal{D})$는 상수라 무관).

### VI의 특징

| 장점 | 단점 |
|---|---|
| 빠름 (경사하강법) | Posterior를 근사 (정확하지 않을 수 있음) |
| 대규모 데이터에 확장 가능 | 분포족 $\mathcal{Q}$ 선택에 따라 품질 달라짐 |
| 딥러닝과 자연스럽게 결합 | 과소추정 경향 (variance underestimation) |

---

## MCMC vs VI 선택 기준

| 상황 | 추천 |
|---|---|
| 정확한 posterior가 중요, 데이터 소규모 | MCMC (NUTS) |
| 대규모 데이터, 속도 중요 | VI |
| 딥러닝 모델 내 불확실성 추정 | VI (또는 5편의 방법들) |
| 탐색적 분석, 모델 비교 | MCMC |

---

## 실용 도구

```python
import pymc as pm

with pm.Model():
    theta = pm.Beta("theta", alpha=1, beta=1)
    obs = pm.Binomial("obs", n=10, p=theta, observed=7)
    trace = pm.sample(1000, tune=1000)  # NUTS 자동 사용
```

---

## 핵심 요약

- Posterior의 분모 적분이 intractable → 두 가지 해법
- **MCMC**: 분포에서 샘플링 — 정확하지만 느림
- **VI**: 최적화 문제로 변환 — 빠르지만 근사
- 현대 베이지안 = 계산 문제를 잘 다루는 것

---

## 다음

[5편]({% post_url 2026-05-29-bayesian-inference-5-uncertainty %})에서는 이 구조가 딥러닝으로 어떻게 이어지는가 — 모델이 "모른다"고 말할 수 있는 메커니즘을 다룹니다.
