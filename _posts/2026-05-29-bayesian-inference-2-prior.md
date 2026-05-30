---
title: "베이지안 추론 2 — 사전분포는 편견인가 지식인가"
date: 2026-05-29 10:00:00 +0900
categories: [Machine Learning]
tags: [bayesian, prior, informative-prior, conjugate, prior-sensitivity, hyperparameter]
math: true
published: false
---

## Prior는 가정을 명시화하는 도구

베이지안에서 가장 많이 받는 비판이 있습니다.

> "prior를 자기 마음대로 잡으면 결론도 마음대로 아닌가?"

반론: **빈도주의도 가정이 있습니다. 다만 숨겨져 있을 뿐입니다.**

- OLS의 가우시안 오차 가정
- MLE의 i.i.d. 가정
- 정규화 계수 $\lambda$를 어떤 값으로 쓸지

베이지안은 그 가정을 prior라는 형태로 명시적으로 드러냅니다.

---

## Prior의 종류

### Flat (non-informative) prior

$$
p(\theta) \propto 1
$$

아무 정보도 없다는 직관적 표현처럼 보이지만, 실제로는 "정보 없음"을 표현하는 것이 생각보다 어렵습니다.  
특히 파라미터를 변환하면 더 이상 균등하지 않을 수 있어, Jeffreys prior 같은 대안이 논의됩니다.

### Weakly informative prior

완전히 균등하지 않지만 과도하게 강하지도 않은 분포입니다.  
예: 회귀계수 $w \sim \mathcal{N}(0, 10^2)$ — "0 근처를 선호하지만 크게 벗어날 가능성도 열어둔다"

현대 베이지안 모델링에서는 이런 weakly informative prior가 자주 권장됩니다.

### Informative prior

실제 사전 지식을 담습니다.

예: 어떤 약의 효과가 기존 문헌에서 평균 0.3, 표준편차 0.05로 알려진 경우:

$$
\theta \sim \mathcal{N}(0.3, 0.05^2)
$$

---

## 켤레 사전분포 (Conjugate Prior)

Prior와 posterior가 같은 분포족에 속하면 **conjugate** 관계라 합니다.

| Likelihood | Conjugate Prior | Posterior |
|---|---|---|
| Binomial | Beta | Beta |
| Poisson | Gamma | Gamma |
| Gaussian (분산 알 때) | Gaussian | Gaussian |
| Multinomial | Dirichlet | Dirichlet |

계산이 닫힌 형태(closed form)로 나와 편리합니다.  
단, 복잡한 모델에서는 conjugate가 존재하지 않아 MCMC/VI가 필요합니다 ([4편]({% post_url 2026-05-29-bayesian-inference-4-mcmc-vi %}) 참고).

---

## 데이터가 많으면 Prior는 흐려진다

Prior가 달라도 데이터가 충분하고 모델이 잘 지정되었다면 posterior는 종종 비슷해집니다.

$$
p_1(\theta \mid \mathcal{D}) \approx p_2(\theta \mid \mathcal{D}) \quad \text{(서로 다른 합리적 prior들)}
$$

반대로 데이터가 적을 때는 prior의 영향이 큽니다.  
이 때 **prior sensitivity analysis** — prior를 바꿔가며 결론이 얼마나 달라지는지 확인하는 작업이 중요합니다.

---

## Prior = Hyperparameter

Prior를 결정하는 파라미터를 **hyperparameter**라 합니다.

$$
\theta \sim \text{Beta}(\alpha, \beta) \quad \leftarrow \alpha, \beta \text{ 가 hyperparameter}
$$

Hyperparameter에도 분포를 씌우면 **hierarchical model**이 됩니다. 이 구조는 데이터가 여러 집단에 걸쳐 있을 때 prior의 강도나 위치를 함께 추정하는 효과를 줍니다.

ML에서 흔히 쓰는 "hyperparameter tuning"과 닿아 있는 언어이지만, 둘이 완전히 같은 절차는 아닙니다.  
베이지안에서는 일부 hyperparameter에 다시 분포를 두는 방식으로 불확실성을 더 직접 모델링할 수 있습니다.

---

## 핵심 요약

- Prior는 **가정을 명시화**하는 도구 — 있고 없고의 문제가 아니라 보이고 안 보이고의 문제
- 데이터가 많으면 prior 영향 감쇠, 적으면 prior가 결론을 크게 좌우
- 좋은 prior는 "아무것도 모른다"가 아니라 **도메인 지식을 적당히 담은** weakly informative

---

## 다음

[3편]({% post_url 2026-05-29-bayesian-inference-3-inductive-bias %})에서는 ML 모델의 아키텍처 선택을 베이지안의 언어로 어떻게 해석할 수 있는지 — inductive bias와 prior의 연결고리를 다룹니다.
