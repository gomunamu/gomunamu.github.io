---
title: "베이지안 추론 3 — Inductive Bias와 Prior는 같은 말이다"
date: 2026-05-29 11:00:00 +0900
categories: [Machine Learning]
tags: [inductive-bias, prior, regularization, gaussian-process, no-free-lunch, architecture, bayesian]
math: true
---

## Inductive Bias란

학습 알고리즘은 학습 데이터만으로는 정답을 특정할 수 없습니다.  
같은 훈련 데이터에 들어맞는 함수는 무한히 많기 때문입니다.

**Inductive bias**는 알고리즘이 데이터와 무관하게 어떤 가설을 선호하는지를 나타냅니다.

> "데이터가 없어도 갖고 있는 가정" = inductive bias = prior

---

## No Free Lunch Theorem

> 모든 태스크에 대해 평균적으로 최선인 학습 알고리즘은 존재하지 않는다.

Inductive bias가 없으면 일반화가 불가능합니다.  
어떤 알고리즘이 잘 작동한다는 것은 그 알고리즘의 inductive bias가 해당 문제의 구조와 맞아떨어진다는 뜻입니다.

베이지안 언어로: **좋은 prior를 가졌다는 뜻.**

---

## ML 모델의 Prior들

### 정규화 = 파라미터에 대한 Prior

[이전 포스팅]({% post_url 2026-05-27-bayesian-regression-map %})에서 다뤘지만, 핵심만 다시:

| 정규화 | Prior | 믿음 |
|---|---|---|
| L2 (Ridge) | $w \sim \mathcal{N}(0, \sigma^2)$ | 가중치는 0 근처에 고르게 분포 |
| L1 (Lasso) | $w \sim \text{Laplace}(0, b)$ | 가중치 대부분은 0, 일부만 크다 |

$\lambda$를 크게 잡는 것 = 좁은 prior = 강한 믿음.

### 커널 함수 = 함수 공간의 Prior

Gaussian Process는 함수에 직접 prior를 씌웁니다.

$$
f \sim \mathcal{GP}(\mu, k)
$$

커널 $k(x, x')$는 두 점의 유사도 — "비슷한 $x$면 비슷한 $f(x)$일 것"이라는 prior.

- RBF 커널: 부드러운 함수를 선호
- 주기 커널: 반복 패턴을 선호

커널 선택 = 함수 공간에서 어떤 해를 선호할지 결정 = prior 설계.

### 아키텍처 = 구조적 Prior

| 아키텍처 | 내재된 Prior |
|---|---|
| CNN | 지역 패턴, 평행이동 불변성이 중요하다 |
| RNN / LSTM | 시간 순서가 중요하다, 최근이 더 중요하다 |
| Transformer | 위치 무관하게 임의의 쌍이 관계를 가질 수 있다 |
| GNN | 그래프 구조의 인접성이 중요하다 |

아키텍처를 선택하는 순간, 어떤 함수 공간을 탐색할지 제한하고 있습니다.  
이것이 **구조적 prior** (structural prior) 또는 **architectural inductive bias**입니다.

---

## 왜 CNN이 이미지에서 잘 되는가

CNN의 inductive bias:
1. 지역성 (locality): 픽셀은 가까운 픽셀과 관계가 있다
2. 평행이동 불변성 (translation equivariance): 고양이는 어느 위치에 있어도 고양이다

이 prior가 자연 이미지의 실제 구조와 잘 맞아떨어지기 때문에 잘 됩니다.  
MLP로도 이미지를 학습할 수 있지만, 더 많은 데이터가 필요합니다 — prior가 약하니까.

---

## 인간의 학습과 같은 구조

인간도 inductive bias를 갖고 태어납니다.

- 얼굴 인식 회로 (신생아도 얼굴을 선호)
- 언어 학습의 보편 문법 (Chomsky)
- 인과관계 추론 성향

이것이 "진화로 형성된 prior"입니다.  
ML 모델의 아키텍처 설계는 이 진화적 prior를 인위적으로 설계하는 과정과 같습니다.

---

## 핵심 요약

- **Inductive bias = prior**: 같은 개념을 ML과 베이지안이 다른 언어로 부름
- 아키텍처 선택, 정규화, 커널 — 모두 prior 설계
- 좋은 모델 = 문제의 구조와 맞는 prior를 가진 모델
- 데이터가 많으면 prior 영향 감소, 적으면 prior(=아키텍처)가 지배적

---

## 다음

[4편]({% post_url 2026-05-29-bayesian-inference-4-mcmc-vi %})에서는 사후 분포를 실제로 어떻게 계산하는가 — MCMC와 Variational Inference를 다룹니다.
