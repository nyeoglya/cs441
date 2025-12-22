- 여기서는 camera model fitting을 예시로 개념을 설명한다.
### extra-1. Fitting Problem
- 소실점(vanishing point) 찾기: 이미지에서 line 찾고, 그것들이 만나는 지점을 찾기. 즉, line estimation(fitting)임.
- image stitching: 이미지 붙이기(estimate homographic transformation)
	- 왼쪽 이미지의 coords.를 오른쪽의 coords.로 변환할 수 있어야 함.
	- 이미지 파노라마에 사용된다.
- model fitting: 모든 AI 모델의 training은 model fitting임. 즉, 이미지를 잘 fit하는 모델을 만드는 것이 중요함.
- challenges: noise, outlier, missing data
### extra-2. Least Square Methods
- 아주 유명하고 많이 사용한다. data points를 많이 가지고 있으면 그걸 잘 근사하는 function을 찾는 방법임.
- outlier에 민감하다는 단점이 있다. 만약 데이터에 이상한 값이 있다면, 그거에 맞춰서 전체 model fitting이 크게 변한다.
	- 이걸 해결하려면 robust error function을 만들거나, RANSAC을 이용하면 된다.
#### extra-2-1. Linear Least Square
- $(x_1,y_1) ,\cdots, (x_n,y_n)$와 같은 데이터가 주어졌을 때, 이를 가장 잘 근사하는 linear function $y=ax+b$의 매개변수 $a,b$를 찾는다.
- 데이터를 모아서, $X[a,b]^T=y$ 꼴로 만들고 이를 푼다.
- objective function (convex)
$$\min \sum_{i=1}^n (ax_i+b-y_i)^2 = \min ||X[a,b]^T-y||^2$$
- global optimum = gradient가 0인 지점이다. 즉, $[a,b]^T=(X^TX)^{-1}X^Ty$
- 한계점: vertical line은 근사를 못한다. 또한, slope가 높은 line은 에러가 매우 커지기 때문에 근사가 어렵다.
#### extra-2-2. Total Linear Least Square
- linear least square가 세로 거리에 아주 민감하기 때문에 나온 방법. 수직 방향의 거리를 이용한다.
- $(x_1,y_1) ,\cdots, (x_n,y_n)$와 같은 데이터가 주어졌을 때, $ax+by+c=0$이 되는 매개변수 $a,b,c$를 찾는 것이 목적이다.
- objective function
$$\min \sum_{i=1}^n (ax_i+by_i+c)^2 = \min ||X[a, b, c]^T||^2$$
- 한계점: $(0,0,0)$이 trivial solution이다(결과가 이게 되어버리면 안됨). 또한, 해가 유일하지 않다. ($a,b,c$ 전체에 상수배를 해도 line equation은 안변하기 때문.)
- 그래서, 아래와 같은 제약조건을 추가한다.
$$||[a,b,c]||=1$$
- 이러한 제한조건이 있는 문제를 푸려면 eigen decomposition을 이용해야 한다. 지금부터는 $a=[a,b,c]$라고 쓴다.
$$||Xa||^2=a^T (X^TX)a = a^T(VDV^T)a=(V^Ta)^TD(V^Ta)$$
- 즉 아래와 같은 새로운 object function이 나온다.
$$\min||Xa||^2=\min b^TDb\quad\text{subejct to}\quad ||b||=1$$
- 여기서 $D=\text{diag}(\lambda_1,\cdots,\lambda_n)$ and $b=V^Ta$이다. 즉, 전체 obj 함수는 $\sum_{i=1}^n b_i^2 \lambda_i$가 된다.
- 이제, $||b||=1$ 만족하면서 obj가 최소가 되도록 해야 한다. 그건 가장 작은 $\lambda$를 찾고 거기에 $b_i$를 몰빵하면 된다. 그러면, 그 eigenvalue에 대응되는 eigenvector만 살아남게 된다.
- 즉, 최적해 $a^*$는 $X^TX$의 가장 작은 eigenvalue에 대응되는 eigenvector이다.
#### extra-2-3. Robust Error Function
- 큰 outlier가 주어져도 linear least square보다 덜 민감하도록 에러 함수를 변경하는 방법도 있다.
- absolute error: $\sum_i |\hat{y_i}-y_i|$
	- 제곱이 아니라 그냥 절댓값만 씌워서 outlier의 영향력을 줄이는 방법
- smoothed $l_1$ loss들도 여러 개 있다. 각각은 비슷한 개형이다. 연구 중에 그들을 하나의 형태로 표현하는 것도 있다.
### extra-3. Random Sample Consensus (RANSAC)
- 많은 수의 outlier를 다루는데 유용한 알고리즘이다. iterative algorithm이고, 모델 추론 시간을 줄이는데 유용하다.
- RANSAC 단계:
	1. 점들의 집합을 랜덤하게 최소로 고른다. (2차원에서 직선 만드려면 점 2개가 있으면 됨. 3차원에서 평면이면 점 3개가 필요)
	2. 결정한 점으로 만들어지는 유일한 초평면을 결정하고 이를 통해 model hypothesis를 만든다(모델 추론). 그 후, 각 점과 error를 계산하고, error가 threshold보다 작은 inlier 점들의 개수를 구한다.
	3. 위의 과정을 정해진 횟수만큼 여러 번 한다. 그러면 여러 model hypothesis가 나오는데, 여기서 inline 점 개수가 가장 많았던 모델을 선택한다.
- 모델의 complexity는 모델의 추론을 위해 필요한 점의 개수에 의존한다. (2차원이더라도 원을 모델로 만들고 싶다면 점이 3개가 필요함)
- 위의 과정을 많이 반복할수록 더 나은 모델이 될 확률이 높아진다.
	- iteration의 횟수를 $N$, 모델의 추론을 위해 결정해야 하는 점의 개수를 $s$라고 하자. outlier 비율 $e$와 noise-free parameter의 추측 확률 $p$에 대해 다음이 성립한다.
	$$(1-(1-e)^s)^N = 1-p \quad \Longrightarrow \quad N = \frac{\log(1-p)}{\log(1-(1-e)^s)}$$
	- 왼쪽 식의 의미는 만들어진 $N$개의 hypothesis model이 전부 noise를 포함하고 있을 확률이 1-p라는 것이다.
- 장점: 간단하고 잘 작동함
- 단점: 많은 iteration이 필요하고, 매개변수의 조정이 많이 필요하다.
