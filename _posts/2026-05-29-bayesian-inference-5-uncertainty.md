---
title: "베이지안 추론 5 — 모델이 모른다고 말할 수 있을 때"
date: 2026-05-29 13:00:00 +0900
categories: [Machine Learning]
tags: [bayesian-deep-learning, uncertainty, epistemic, aleatoric, dropout, bnn, active-learning, ood]
math: true
---

## 일반 딥러닝의 문제

잘 훈련된 신경망도 본 적 없는 입력에 대해 높은 확신으로 틀린 답을 냅니다.

> 모델이 "0.97 확률로 고양이"라고 말할 때, 그 0.97을 믿어도 되는가?

일반 신경망은 **점추정** (point estimate) — 가중치 $w$에 대해 하나의 값을 줍니다.  
불확실성을 표현하는 구조 자체가 없습니다.

---

## 불확실성의 두 종류

### Aleatoric Uncertainty (우연적 불확실성)

데이터 자체의 노이즈에서 비롯된 불확실성.  
데이터를 아무리 많이 모아도 줄어들지 않습니다.

예: 흐릿한 이미지, 센서 노이즈, 레이블 모호성

### Epistemic Uncertainty (인식론적 불확실성)

모델이 충분히 학습하지 못해서 생기는 불확실성.  
데이터를 더 모으면 줄어들 수 있습니다.

예: 훈련 데이터에 없는 영역, 소수 클래스

| | Aleatoric | Epistemic |
|---|---|---|
| 원인 | 데이터 노이즈 | 모델의 무지 |
| 데이터 추가 시 | 줄어들지 않음 | 줄어듦 |
| 베이지안 표현 | Likelihood의 분산 | Posterior의 분산 |

---

## Bayesian Neural Network (BNN)

가중치 $w$를 점추정 대신 **분포**로 표현합니다.

$$
p(w \mid \mathcal{D}) \propto p(\mathcal{D} \mid w)\, p(w)
$$

예측 시에는 posterior로 적분:

$$
p(y \mid x, \mathcal{D}) = \int p(y \mid x, w)\, p(w \mid \mathcal{D})\, dw
$$

이 적분이 epistemic uncertainty를 자연스럽게 포착합니다.

문제: 수백만 개의 가중치에 대해 posterior를 구하는 것은 계산적으로 매우 어렵습니다.

---

## Dropout = Approximate BNN

Gal & Ghahramani (2016)의 발견:

> **테스트 시에도 dropout을 켜두고 여러 번 예측하면, 이는 BNN의 approximate posterior sampling과 동치다.**

```python
# 훈련 시
model.train()
pred = model(x)  # dropout 켜짐

# 테스트 시 (불확실성 추정)
model.train()  # eval() 아님!
preds = [model(x) for _ in range(100)]
mean = torch.stack(preds).mean(0)
variance = torch.stack(preds).var(0)  # epistemic uncertainty
```

분산이 크면 모델이 해당 입력에 대해 불확실하다는 신호입니다.

---

## 모델 지능화의 메커니즘

베이지안 관점에서 모델이 "지능적"이 된다는 것의 의미:

### 1. 아는 것과 모르는 것을 구분

훈련 데이터 분포 안 (in-distribution): 낮은 uncertainty  
분포 밖 (out-of-distribution): 높은 uncertainty

### 2. 데이터로 Prior를 갱신

새 데이터 → posterior 갱신 → 다음 prior  
이 루프가 반복될수록 모델의 믿음이 정교해집니다.

### 3. 어디서 더 배워야 할지 안다 — Active Learning

$$
x^* = \arg\max_x \text{Uncertainty}(x)
$$

불확실성이 높은 데이터를 우선 수집 → 가장 효율적인 학습.

---

## 실용적 활용

| 상황 | 활용 |
|---|---|
| 의료 진단 | "이 케이스는 모델이 불확실합니다 — 전문의 확인 필요" |
| 자율주행 | OOD 입력 감지 → 안전한 행동으로 전환 |
| 금융 | 예측 불확실성을 리스크 추정에 반영 |
| 강화학습 | Epistemic uncertainty = 탐험 가치 |

---

## 전체 시리즈 연결

```
1편: 데이터로 분포를 갱신한다
2편: Prior = 명시화된 가정
3편: Inductive bias = Prior (ML과 베이지안은 같은 말)
4편: Posterior 계산 = MCMC / VI
5편: 불확실성 정량화 = 모델의 지능화
```

이 다섯 가지 시각이 연결되면:

> **ML 모델을 설계한다** = prior를 정의한다  
> **모델을 학습시킨다** = posterior를 추정한다  
> **모델이 지능적이다** = 불확실성을 알고 표현한다

---

## 핵심 요약

- 일반 신경망은 불확실성을 표현하지 못함
- BNN: 가중치에 분포 → 예측 불확실성 정량화
- Dropout: 구현이 간단한 approximate BNN
- Epistemic uncertainty = 모델이 모르는 것 = 데이터로 줄일 수 있는 것
- 모델 지능화 = 아는 것/모르는 것을 구분하는 능력
