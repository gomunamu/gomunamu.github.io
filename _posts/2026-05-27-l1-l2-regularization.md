---
title: "L1·L2 규제: Lasso와 Ridge의 원리부터 실전까지"
date: 2026-05-27 09:00:00 +0900
categories: [Machine Learning]
tags: [regularization, lasso, ridge, l1, l2, feature-selection, overfitting, python, sklearn]
math: true
---

## 서론

모델을 학습하다 보면 훈련 데이터에는 잘 맞지만 새로운 데이터에는 맥을 못 추는 **과적합(overfitting)** 문제를 자주 마주칩니다.  
규제(regularization)는 이 문제를 다루는 대표적인 기법으로, 손실 함수에 **패널티 항**을 더해 모델이 지나치게 복잡해지는 것을 억제합니다.

이 포스팅에서는 가장 많이 쓰이는 두 가지 규제 방식을 정리합니다.

1. **L2 규제 (Ridge)** — 가중치를 전체적으로 줄인다
2. **L1 규제 (Lasso)** — 가중치를 0으로 만들어 특성을 선택한다
3. **왜 L1은 sparse하고 L2는 smooth한가** — 기하학적·미분적 관점
4. **Elastic Net** — 둘을 섞으면?
5. **실전 코드**

---

## 1. 규제가 없으면 어떤 일이 일어나는가

선형 회귀의 목표는 잔차 제곱합(RSS)을 최소화하는 계수 $\mathbf{w}$를 찾는 것입니다.

$$
\mathcal{L}_{\text{OLS}} = \sum_{i=1}^{n} \left( y_i - \mathbf{w}^\top \mathbf{x}_i \right)^2
$$

특성 수가 많거나 데이터가 적으면, 모델은 훈련 데이터의 **노이즈까지 외워버립니다.** 결과적으로 일부 $w_j$가 매우 크게 추정되어 분산이 폭발합니다.

> **편향-분산 트레이드오프**: 규제를 강하게 주면 편향은 늘지만 분산은 줄어들어, 전반적인 테스트 오차가 감소할 수 있습니다.

---

## 2. L2 규제 — Ridge

### 2.1 목적 함수

$$
\mathcal{L}_{\text{Ridge}} = \sum_{i=1}^{n} \left( y_i - \mathbf{w}^\top \mathbf{x}_i \right)^2 + \lambda \sum_{j=1}^{p} w_j^2
$$

패널티 항 $\lambda \|\mathbf{w}\|_2^2$이 **모든 가중치의 제곱합**을 억제합니다.  
$\lambda$가 클수록 계수가 0에 가깝게 수축(shrinkage)됩니다.

### 2.2 닫힌 형태 해 (Closed-form solution)

행렬로 표현하면 Ridge의 해는 해석적으로 구할 수 있습니다.

$$
\hat{\mathbf{w}}_{\text{Ridge}} = \left( X^\top X + \lambda I \right)^{-1} X^\top \mathbf{y}
$$

OLS의 해 $(X^\top X)^{-1} X^\top \mathbf{y}$와 비교하면, 대각 행렬 $\lambda I$가 더해져 **역행렬이 항상 존재**합니다. 다중공선성(multicollinearity) 문제도 자연스럽게 완화됩니다.

### 2.3 핵심 특성

| 항목 | 내용 |
|---|---|
| 패널티 | $\lambda \sum w_j^2$ (L2-norm의 제곱) |
| 계수 거동 | 0에 가깝게 수축하지만, **정확히 0이 되진 않음** |
| 특성 선택 | 불가 (모든 특성이 모델에 남음) |
| 다중공선성 | 강건 |
| 미분 가능 | 모든 점에서 가능 |

---

## 3. L1 규제 — Lasso

### 3.1 목적 함수

$$
\mathcal{L}_{\text{Lasso}} = \sum_{i=1}^{n} \left( y_i - \mathbf{w}^\top \mathbf{x}_i \right)^2 + \lambda \sum_{j=1}^{p} |w_j|
$$

패널티가 $\|\mathbf{w}\|_1$, 즉 **절댓값의 합**으로 바뀝니다.

### 3.2 핵심 특성

| 항목 | 내용 |
|---|---|
| 패널티 | $\lambda \sum \|w_j\|$ (L1-norm) |
| 계수 거동 | 일부 계수가 **정확히 0**이 됨 |
| 특성 선택 | 가능 (희소 모델 생성) |
| 다중공선성 | 상관된 특성 중 하나만 선택하는 경향 |
| 미분 가능 | $w_j = 0$에서 미분 불가 → subdifferential 사용 |

---

## 4. 왜 L1은 0을 만들고 L2는 만들지 않는가

### 4.1 기하학적 관점

제약 최적화 문제로 바꿔 쓰면 이해하기 쉽습니다.

- **Ridge**: $\|\mathbf{w}\|_2^2 \le t$ 조건 하에 RSS 최소화 → 제약 영역이 **구(ball)**
- **Lasso**: $\|\mathbf{w}\|_1 \le t$ 조건 하에 RSS 최소화 → 제약 영역이 **다이아몬드(마름모)**

RSS의 등고선(타원)이 제약 영역과 처음 만나는 점이 해입니다.

```
L2 (Ridge)          L1 (Lasso)

   RSS 등고선            RSS 등고선
      ~~~                   ~~~
    (   ●  )              (  ●   )
   (  구형  )            (다이아몬드)
    (     )               /  \
                         /    \
                        ◆ ← 꼭짓점에서 만남
```

L1의 다이아몬드는 **꼭짓점(vertex)이 축 위**에 있습니다. 등고선이 꼭짓점에서 접하면 해당 좌표는 정확히 0이 됩니다. 고차원에서는 꼭짓점이 많아질수록 희소해집니다.

L2의 구는 꼭짓점이 없으므로, 등고선이 어디서 접하든 두 좌표가 동시에 0이 될 가능성은 매우 낮습니다.

### 4.2 미분 관점 (Soft Thresholding)

1차원 단순화 문제로 생각해 봅시다. 잔차 제곱합의 최솟점이 $\hat{w}_{\text{OLS}} = c$라 할 때:

**Ridge** 업데이트:

$$
\hat{w}_{\text{Ridge}} = \frac{c}{1 + \lambda}
$$

$c$가 얼마든 $\hat{w}$는 0이 되지 않고, $c$에 비례해 수축합니다.

**Lasso** 업데이트 (Soft Thresholding):

$$
\hat{w}_{\text{Lasso}} = \text{sign}(c) \cdot \max\left(|c| - \frac{\lambda}{2},\ 0\right)
$$

$|c| < \lambda/2$이면 **정확히 0**이 됩니다. 절댓값이 작은 계수는 통째로 잘려나갑니다.

이것이 Lasso가 자동으로 **특성 선택(feature selection)** 을 수행하는 이유입니다.

---

## 5. $\lambda$ 값의 역할

$$
\lambda = 0 \Rightarrow \text{규제 없음 (OLS)}
\quad\quad
\lambda \to \infty \Rightarrow \text{모든 계수} \to 0
$$

| $\lambda$ | 편향 | 분산 | 적합 |
|---|---|---|---|
| 작다 | 낮음 | 높음 | 과적합 위험 |
| 적절 | 중간 | 중간 | 최적 |
| 크다 | 높음 | 낮음 | 과소적합 |

최적 $\lambda$는 **교차 검증(cross-validation)** 으로 탐색합니다.

---

## 6. Elastic Net — L1과 L2의 혼합

Lasso는 상관된 특성이 있을 때 그 중 하나만 선택하고 나머지를 버리는 경향이 있습니다. Elastic Net은 두 패널티를 선형 결합합니다.

$$
\mathcal{L}_{\text{ElasticNet}} = \text{RSS} + \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2
$$

또는 혼합 비율 $\rho \in [0, 1]$로 표현하면:

$$
\mathcal{L} = \text{RSS} + \lambda \left[ \rho \|\mathbf{w}\|_1 + \frac{1-\rho}{2} \|\mathbf{w}\|_2^2 \right]
$$

- $\rho = 1$: Lasso
- $\rho = 0$: Ridge
- $0 < \rho < 1$: 그룹 내 상관 특성을 함께 선택하면서 희소성도 확보

---

## 7. 비교 요약

| | **Ridge (L2)** | **Lasso (L1)** | **Elastic Net** |
|---|---|---|---|
| 패널티 | $\|\mathbf{w}\|_2^2$ | $\|\mathbf{w}\|_1$ | $\rho\|\mathbf{w}\|_1 + \frac{1-\rho}{2}\|\mathbf{w}\|_2^2$ |
| 계수 = 0? | 아니오 | 예 | 예 (일부) |
| 특성 선택 | 불가 | 가능 | 가능 |
| 다중공선성 | 강건 | 취약 (하나만 선택) | 강건 |
| 닫힌 해 | 존재 | 없음 (iterative) | 없음 (iterative) |
| 주 용도 | 계수 안정화 | 희소 모델 | 균형 |

---

## 8. 실전 코드 (Python / sklearn)

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# 규제 전 스케일링은 필수
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

# Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
print("0인 계수 수:", np.sum(lasso.coef_ == 0))

# Elastic Net
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train_scaled, y_train)

# 교차 검증으로 alpha 탐색
from sklearn.linear_model import RidgeCV, LassoCV
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5)
ridge_cv.fit(X_train_scaled, y_train)
print("최적 alpha:", ridge_cv.alpha_)
```

> **스케일링 주의**: 규제는 계수의 크기에 직접 페널티를 주므로, 특성 간 단위가 다르면 결과가 왜곡됩니다. 반드시 `StandardScaler`로 정규화한 뒤 규제를 적용하세요.

---

## 9. 언제 무엇을 쓸까

| 상황 | 추천 |
|---|---|
| 모든 특성이 유의미하다고 생각될 때 | Ridge |
| 특성이 많고 일부만 관련 있을 것 같을 때 | Lasso |
| 특성 간 상관이 높고 희소성도 원할 때 | Elastic Net |
| 탐색 단계에서 잘 모를 때 | Elastic Net (범용) |

---

## 마치며

L1과 L2 규제는 단순히 "과적합을 막는 테크닉"을 넘어, **어떤 종류의 해를 선호하는가**라는 사전 가정(prior)을 모델에 심는 행위입니다.

- L2 = 가우시안 사전 분포 → 계수가 전체적으로 작기를 기대
- L1 = 라플라스 사전 분포 → 계수 대부분이 0이기를 기대

베이즈 관점에서 보면 규제는 MAP(Maximum A Posteriori) 추정과 동치입니다. 이 연결은 나중에 베이지안 회귀 포스팅에서 더 자세히 다루겠습니다.
