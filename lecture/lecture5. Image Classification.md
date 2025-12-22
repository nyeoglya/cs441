### 5-1. Image Classification
- scene recognition: 주어진 이미지의 배경을 분류하기
- object classification: 이미지 내의 물체를 분류하기
- 2가지 단계로 나뉜다.
	- 이미지를 numerical form으로 변환하기
	- classification function을 만들어, 변환한 numerical form을 category label로 매핑하기.
- classical image classification: hand-crafted 방법으로 이미지를 numerical form으로 변환. bag of words(BoW)를 사용해서 histogram을 만든다. 우리가 이미했던 거임.
### 5-2. Classification Algorithms
- classifier: image descriptor(이미지의 벡터 표현)를 category label로 대응하는 함수
- 2가지 단계를 통해 구성한다.
	- training: label된 데이터 가지고 예측 함수 $f$ 추론하기. 학습을 위해서 loss 함수를 정의한다.
	- testing: 학습된 $f$를 실제 데이터에 넣고 테스트한다.
- classifier는 다음과 같은 종류들이 있다.
	- nearest neighbor
	- naive bayes
	- support vector machine(=SVM)
	- boosting
	- decision tree
	- random forest
	- deep neural network
#### 5-2-1. Several Types of Classification Model
- 뷴류 모델을 구분하는 기준이 여러 개가 있다.
- linear vs non-linear: 데이터 분류 모델이 선형적인지 비선형적인지 분류함.
	- non-linear는 더 많은 파라미터가 있기 때문에 데이터가 많이 필요하다.
- generative vs discriminative
	- generative: likelihood와 prior 기반
		- 생성 모델은 데이터 X와 레이블 Y의 결합 확률 분포를 모델링하는 것을 목적으로 한다.
		- 근데, 여기서는 뭔가를 생성하는 건 아니고, 이미지 특징을 likelihood 모델로 추출하는 역할임.
		- prior: 특정 클래스가 얼마나 자주 등장하는지 비율을 알려줌.
		- likelihood: 클래스가 주어졌을 때, 데이터의 특성이 갖는 분포를 알려줌.
		- 위의 두 정보를 이용해, 특정 데이터 X가 주어졌을 때, 어떤 클래스가 가장 X를 생성했을 확률이 높은지를 설명하는 것이 목적.
		- 실제 distribution을 추론하는 것이 목적인 naive-bayes classifier이 대표적인 예시.
	- discriminative: posterior 기반
		- 데이터가 어디에 속하는지 직접적으로 모델링해서 데이터를 잘 분리하는 경계를 찾는 게 목적임.
		- posterior: 데이터 X가 주어졌을 때, Y가 나타날 직접적인 확률
		- logistic regression이 대표적인 예시. 이건 선 가르기만 하고 정확한 distribution은 관심을 주지 않음. 즉, regression 모델이 아니고, classification 모델임.
		- 그냥 정확한 경계선을 찾는데만 관심이 있기 때문에 더 효율적이다.
	- 기본적으로 두 종류의 모델 다 확률에 의존하지만, discriminative model 중에서는 확률적 해석 없이 바로 label을 추론하는 종류도 있음. LDA나 SVM이 대표적인 예시.
- K-nearest neighbor
	- 모든 데이터가 classified되어 있고, 데이터베이스에 저장된다.
	- 새 점이 들어오면, 가장 가까운 k개 점과 비교해서 dominant한 label을 선택한다.
	- 애초에 학습이 필요없으며, 데이터가 많으면 계산이 많아서 비효율적임.
### 5-3. Support Vector Machines
- concept of SVM: 가장 좋은 hyperplane을 그리기.
- margin을 maximize하는 것이 가장 좋다는 가정을 하고, 이를 찾는 것을 목적으로 한다.
- 강력한 분류 방법이지만, multi-class classification으로 확장하는 직접적인 방법이 없다.
#### 5-3-1. Linear SVM, Separable Case
- 우선 데이터가 하나의 hyperplane으로 완전히 분리가 가능한 상태 가정하자.
- 주어진 데이터: $(x_1,y_1), \cdots, (x_n,y_n)$. 여기서 $x_i \in \mathbb{R}^k$, $y_i \in \{-1,1\}$이다.
- hyperplane $w^\intercal x + b$에 대해, 거리와 margin을 각각 계산할 수 있다. 그 중에서 각 class에서 가장 거리가 가까운 점을 하나씩 총 2개를 고른다. 두 거리의 합이 최대가 되도록 하는 것이 목적이다. (max-margin solution)
$$\text{argmax}_{w,b}{\left(\min_i \frac{|w^T x_i+b|}{||w||}\right)}=\text{argmax}_{w,b}{\left(\frac{1}{||w||} \min_i [y_i(w^T x_i+b)]\right)}$$
- 이건 풀기 복잡하기 때문에, 내부 margin 값이 1 이상이 되도록 파라미터를 조정하여 min 부분을 1로 만들며 제거한다. 그렇게 해서, 바깥의 argmax만 계산(뒤집어서 argmin)한다.
$$\text{argmin}_{w,b}\frac{1}{2}||w||^2\quad \text{subject to} \quad y_i(w^\intercal x_i +b) \geq 1 \;\; \forall i$$
- 이걸 dual form(KKT condition)으로 변형하고, 최적화한다. (여기서는 최적화를 다루지 않아서 생략)
$$
\max_\lambda \min_{w,b} L(w,b,\lambda),\quad\text{where}\quad L(w,b,\lambda)=\frac{1}{2}||w||^2-\sum_{i=1}^n \lambda_i \{y_i (w^T x_i + b)-1\}
$$
	- 여기서 $\lambda_i \ge 0$이며, 이는 largrange multiplier이다.
	- 직관적인 설명: 만약 데이터가 조건을 위배하면, sum 부분이 음수가 되어 전체적으로 $L$을 키운다.
	- 즉, 조건을 위배하지 않는 방향으로 학습이 진행된다. 제약조건을 손실 함수에 집어넣는 많이 쓰는 방법.
- lagrange multiplier: $\text{argmin}_{x,y} f(x,y) \quad \text{subject to} \quad g(x,y) \geq 0$ 
	- lambda를 최대화한다고 해놓은 건, lambda의 영향력을 키워서 loss 함수가 올바르지 않은 값에 더 민감하게 반응하게 하려는 목적이다.
- 이제, dual form을 w와 b에 대해 각각 편미분해서 0이 되는 지점을 찾아서 대시 대입하자. 그러면, $\min_{w,b}$가 사라지면서 $\lambda$에만 의존하는 $\max_\lambda$ 문제가 된다.
$$
\max_\lambda \sum_{i=1}^n \lambda_i - \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j y_i y_j x_i^T x_j\quad\text{subject to}\quad \lambda_i \ge 0 \;\&\;\sum_i \lambda_i y_i = 0
$$
- 이 문제를 quadratic programming이라고 한다. QP solver에 넣으면 그냥 해가 나온다.
- 여기서 몇몇 값들은 $\lambda$가 0이다. 이때, $\lambda>0$인 벡터를 support vector라고 부른다.
	- support vector는 경계에 가장 가까이 있는 데이터포인트들로, margin을 이용한 학습에 기여하게 된다.
	- support vector가 아닌 데이터들($\lambda = 0$)은 학습에 기여하지 않는다.
	- $L$을 w에 대해 편미분하면, $w^*=\sum_i \lambda_i y_i x_i$임을 명심하자. 이는 $w$가 support vector에 의해서만 결정됨을 설명한다. 
	- 또한, bias인 $b$도 모든 support vector $x_i$에 대해, $y_i ({w^*}^Tx_i + b) = 1$임을 이용하면 $b=\frac{1}{n_s}\sum_{i\in S}(y_i - {w^*}^Tx_i)$로 유일하게 결정할 수 있다.
#### 5-3-2. Linear SVM, Non-separable Case
- slack variable $\xi_i \geq 0$를 도입한다. 이 variable은 실제 값이 어긋나는 정도를 측정한다. 위의 variable을 추가하여 아래의 새 obj 함수를 만든다.
$$
\text{argmin}_{w,b} \frac{1}{2} ||w||^2 + C \sum_{i=1}^n \xi_i\quad\text{subject to}\quad y_i(w^Tx_i+b)\ge1-\xi_i\;\&\;\xi_i\ge 0
$$
- 여기서 C를 조절해서 데이터가 많이 오염된 경우, 적게 오염된 경우를 나눌 수 있다.
- 식을 정리하면 아래와 같다.
$$
\text{argmin}_{w,b} \frac{1}{2} ||w||^2 + C \sum_{i=1}^n \max\{0,1-y_i(w^Tx_i+b)\}
$$
#### 5-3-3. Non-Linear SVM
- kernel method: linear하게 구분이 불가능한 데이터의 경우, 새로운 차원을 만들어서 입력 데이터 차원을 늘린다.
- $x \to \phi(x)$로 mapping한다고 하자. 그러면, $f(x)=\sum_{i=1}^n \lambda_i^* y_i \phi(x_i)^\intercal \phi(x) + b$
	- 여기서 $k(x_i,x)=\phi(x_i)^\intercal\phi(x)$라고 하고, kernel이라고 부른다.
	- 정확한 mapping function $\phi$ 자체는 몰라도, $k$만 알면 된다.
	- kernel은 mapping 함수의 결과를 inner product한 것이라고 생각할 수 있다.
- kernel function은 하나의 distance metric이다. 이것이 SVM이 k-mean의 더 고차원적인 버전이라는 것에 대한 또 하나의 이유이다.
- 자주 사용하는 kernel은 linear, polynomial, gaussian, histogram intersection 등이 있다.
- obj. 함수는 SVM obj. 함수의 dual form에서 $x_i^Tx_j$에 mapping function을 적용하면 된다. 정리하면, 다음과 같다. 이제, 아래의 식에서 같은 optimization을 수행하면 된다.
$$
\max_\lambda \sum_{i=1}^n \lambda_i - \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j y_i y_j k(x_i,x_j)
$$
- $\phi$를 정확하게 모르면 inference 단계에서 input이 주어졌을 때, 각각을 전부 support vector와 함께 kernel에 넣어서 계산해야 한다.
- 즉, support vector 개수가 많으면 계산 속도가 느려진다는 단점이 있다.
	- 원래 linear SVM에서는 feature의 수에 계산량이 비례했다.
	- non-linear는 support vector의 수에 비례한다.
#### 5-3-4. Hyper-Parameter Tuning
- SVM은 여러 개의 하이퍼파라미터를 선택해야 한다. 이때, 어떻게 올바른 hyperparameter를 결정할 수 있을까?
- 가장 naive한 방법: training error를 가장 줄이는 value를 선택한다.
	- overfitting의 문제가 있음.
- 더 나은 방법: cross validation.
	- 우선, 데이터를 균등하게 K개로 쪼갠다.
	- 다양한 하이퍼파리미터 세트를 만들어놓고, K-1개로 학습한다.
	- 학습된 결과를 나머지 1개로 평가한다.
	- 위의 단계를 K번 반복하고 평균을 낸다.
	- 평균 오차가 가장 적은 하이퍼파라미터 세트가 가장 좋은 세트임.
#### 5-3-5. Multi-Class SVM
- 다중 클래스 분류. $y_i$가 3개 이상의 값을 가진다. 총 K개 클래스가 있다고 하자.
- one-versus-all(OvA)
	- 각 클래스에 대해, 해당 클래스만 긍정, 나머지 전체를 부정으로 하여 하나의 SVM 학습한다.
	- K개 SVM이 만들어진다.
	- 각 SVM에 넣고 돌린 다음에 가장 높은 값을 반환하는 것을 최종 분류 결과로 선택한다.
- one-versus-one(OvO)
	- 가능한 모든 pair에 대해 분류기를 학습시킨다.
	- $K(K-1)/2$개의 SVM이 생긴다.
	- 새로운 입력 데이터에 대해 모든 SVM을 적용한다. 각 SVM은 두 클래스 중 하나에 투표(vote)한다.
	- 가장 많은 투표를 받은 클래스를 최종 결과로 선택한다. (vote scheme)
