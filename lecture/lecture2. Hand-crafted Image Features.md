### 2-1. Basics of Image Processing
- 모든 이미지 처리는 convolutional operation으로 처리된다. 기존의 convolution과는 약간 다르게 생겼지만, 지금부터는 이걸 convolution이라고 부른다.
$$g=f * h \quad \text{where} \quad g(x,y)=\sum_{u,v}f(x-u,y-v)h(u,v)$$
- 주어진 이미지에, convolution kernel을 겹쳐가며 convolution 연산을 한다. kernel은 local structure를 잡아내는 역할을 수행한다.
- 연산의 결과도 이미지(matrix)이다.
- 이 연산은 commutative, associative, distributive하고, linear filter이다.
#### 2-1-1. Smoothing Filter
- 대표적인 예시 중 하나이다. 이미지를 전체적으로 smooth하게 만든다.
	- K-mean kernel
$$\frac{1}{K^2} \begin{pmatrix}1 & \cdots & 1 \\ \vdots & & \vdots \\ 1 & \cdots & 1\end{pmatrix}$$
	- gaussian kernel: 중심부 픽셀을 더 집중해서, smoothing하면서도 중심 색상의 정보를 더 뽑아내는 용도.
	- bilinear: 중심부 픽셀을 더 집중해서, smoothing하면서도 중심 색상의 정보를 더 뽑아내는 용도
- 이미지의 노이즈를 감소시키려는 목적으로 사용한다. 또한, 다른 작업 전(테두리 감지 등)에 preprocessing 목적으로 많이 쓴다.
#### 2-1-2. Gradient Filter
- gradient는 보통 연속함수에서 정의한다. 특정 점에서의 기울기로.
- 디지털 데이터는 연속적이지 않기 때문에, 미분을 계산하는 것이 불가능하다. 그러나, 연속된 두 픽셀의 차이를 계산함으로써 기울기를 근사하는 것은 가능하다. (x, y 방향)
- 아래는 y방향 필터의 가장 기본적으로 가능한 형태이다.
$$\begin{pmatrix} -1 \\ 0 \\ 1\end{pmatrix}$$
- 그러나, 실제로는 아래같은 필터를 이용한다. (아래는 x-derivative filter)
$$\frac{1}{8} \begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1\end{pmatrix}$$
- x-derivative filter는 vertical line을 감지하는데 이용하고, y-derivative filter는 horizontal line을 감지하는데 이용한다.
#### 2-1-3. Padding
- convolution을 계속 적용하면 결국에는 크기가 1x1로 줄어든다.
- 이에, 여러가지 방법으로 이미지 크기를 유지시키기 위해 픽셀들을 테두리에 더한다.
### 2-2. Local Image Feature
- 이미지 내에서 관심이 가는 영역.
- 이미지 표현법
	- Image-level descriptor: bag of visual words
	- object appearance modeling: 외부적인 모습을 표현하고 분석. pose나 가려짐에 영향을 받지 않아야 한다.
- matching: 한 대상을 여러 위치에서 찍고, 3D로 복원.
- 대체 무엇이 좋은 local feature일까???
	- 다른 것들과 혼동할 정도로 너무 흔하면 안된다. unique할 수록 좋은 feature임.
	- saliency: 흥미로운 파트를 담고 있어야 됨.
	- locality: 충분히 작은 영역이어야 함.
	- repeatability: 같은 특징이 다른 이미지에서 계속 발견이 되어야 됨. (기하적, 이미지적 변형에 무관하게) (중요한 특징임)
#### 2-2-1. Edge
- edge는 구분되는 두 영역의 경계이다. gradient filter를 통해 얻을 수 있다.
- texture/depth나 표면 방향의 불연속에 의해 발생한다.
- discontinuity는 표면 normal vector 방향이 바뀌면 생기는 거임.
	- reflectance discontinuity: 물체 표면에 반사(거울 내부에 방이 비치면 보임).
	- illumination discontinuity: 빛이나 그림자 때문에 생김.
	- reflectance, illumination discontinuity는 실제로는 edge는 아닌데, 마치 edge인 것처럼 보임.
- edge detection
	1. 일단 smoothing 한다. 노이즈가 edge로 오탐지될 확률이 높기 때문.
	2. gradient magnitude를 계산한다. x-derivative, y-derivative 필터링 후에, $\sqrt{I_x^2+I_y^2}$ 계산.
	3. non-max suppression: 주변과 비교해서 local max만 남긴다.
	4. thresholding: 특정 값보다 낮으면 다 없앤다.
- 이때, edge detection 처음 두 과정은 conv.이기에, 다음과 같이 효율 높일 수 있다.
$$I_x=D_x * (G * I)=(D_x * G) * I$$
- gradient filter를 1x3과 3x1가 아니라 3x3을 써야 미리 연산이 된다.
- 여기서 $G$가 gaussian이고, $D_x$가 derivative이다. $D_x * G$를 derivative of gaussian(DoG)라고 부른다.
#### 2-2-2. Corner
- contour가 만나는 지점이다. 근방의 점들이 edge detection에 영향을 받게 됨.
- harris corner detection
	- corner는 영역(=patch)을 조금만 옮겨도 차이가 두드러짐을 이용한다.
	- edge는 선분 방향으로는 차이가 적다.
	- flat region은 모든 방향에 대해 차이가 없다.
- "이미지가 grayscale이라고 가정"하면, $(u,v)^T$만큼의 shift에 대한 intensity의 변화는 다음과 같다.
$$E(u,v)=\sum_{x,y}w(x,y)\{I(x+u,y+v)-I(x,y)\}^2$$
	- $w(x,y)$: window function. 가중치 함수임.
		- 가령, 이 부분을 gaussian filter로 잡으면 바깥 쪽을 좀 덜 고려하게 된다.
	- $I(x+u,y+v)$: shifted intensity(=gray scale). 이동된 영역의 강도.
	- $I(x,y)$: intensity. 기존 이미지 강도.
- 위 식을 first-order taylor expansion으로 근사해서 정리하면 아래와 같다.
$$E(u,v) \approx \begin{pmatrix}u \\ v\end{pmatrix} ^ T \left( \sum_{x, y} w(x,y) \begin{pmatrix}I_x(x,y)^2 & I_x(x,y)I_y(x,y) \\ I_x(x,y)I_y(x,y) & I_y(x,y)^2\end{pmatrix} \right) \begin{pmatrix}u \\ v\end{pmatrix}$$
- 중간 부분은 $(u,v)$에 무관한 값이라 그냥 하나의 행렬 $M$으로 뭉쳐버리면 된다.
- 이제, $(u,v)^T$를 unit vector라 가정하자. 그러면, $(u,v)^T=a_1x_1+a_2x_2\quad\text{where}\quad a_1^2+a_2^2=1$이다. M이 대칭 행렬이라 대각화 가능하고, 이에 orthonormal한 eigenvector인 $x_1, x_2$를 선택할 수 있다. corner point는 $E(u,v)$가 항상 커야 되기 때문에, $\min_{u,v}E(u,v)$가 커야 한다.
$$
\begin{align*}
	\min_{u,v}E(u,v) &\approx \min_{u,v} \begin{pmatrix}u \\ v\end{pmatrix}^T M \begin{pmatrix}u \\ v\end{pmatrix}\\
	&= \min_{a_1,a_2}(a_1x_1+a_2x_2)^T M (a_1x_1+a_2x_2)\\
	&= \min_{a_1,a_2}(a_1^2x_1^TMx_1 + a_2^2x_2^TMx_2)\\
	&= \min_{a_1,a_2}(a_1^2 \lambda_1 + a_2^2 \lambda_2 )\\
	&= \lambda_2
\end{align*}
$$
- 마지막은 WLOG이다. $a_1$이 더 크다고 가정하면 $\lambda_2$에 몰빵하면 된다.
- 즉, 결론적으로 corner가 되려면, M의 eigenvalue 중 작은 것이 커야 한다.
- 그런데, eigenvalue를 구하는 건 시간이 오래 걸리는 일이기 때문에, 아래의 식을 이용한다.
$$R = \det M - k \cdot (\text{tr} M)^2 = \lambda_1 \lambda_2 - k \cdot (\lambda_1 + \lambda_2)^2$$
- 여기서 $k$는 적당히 잘 잡은 값이고, 두 eigenvalue가 크면 $R$도 커진다. 즉, 이 값에 threshold를 정해서 적당히 제거한다.
- 마지막으로, non-max suppression로 특이 케이스를 제거한다.
	- 코너 후보 근처는 전부 높은 R 값을 가지기 때문에 가장 강한 응답 1개만을 남긴다.
#### 2-2-3. Blobs
- blob는 주변에 비해 밝거나 어두운 부분.
- blob의 탐지 방법
	- 입력 이미지 smoothing
	- laplacian of gaussian / difference of gaussian 적용
	- optimal scale / orientation parameter 찾기
- laplacian of gaussian: $\nabla^2 G$
	- convolution kernel은 우리가 찾고자 하는 모양과 유사해야 한다. 이것이 gaussian filter를 blobs에 사용하는 이유다. (모양이 비슷함)
		- 이건 edge detection에 $\nabla G$를 사용하는 이유이기도 하다.
- Laplace operator in general form:
$$\nabla^2 = \left[ \frac{\partial}{\partial x_1}, \cdots, \frac{\partial}{\partial x_N} \right] \left[ \frac{\partial}{\partial x_1}, \cdots, \frac{\partial}{\partial x_N} \right]^T = \sum_{n=1}^N \frac{\partial^2}{\partial x_n^2}$$
- 2D isotropic Gaussian distribution (mean = zero):
$$G(x,y,\sigma)=\frac{1}{2\pi \sigma^2}\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)$$
- Laplacian of 2D isotropic gaussian distribution(=LoG):
$$\nabla^2 G(x,y,\sigma) = \frac{x^2+y^2-2\sigma^2}{\sigma^4} G(x,y,\sigma)$$
	- 위는 continuous하기 때문에 실제 필터는 discrete하게 변환해서 쓴다.
- LoG에서 $\sigma$가 커질수록 큰 blob을 인식하는데 유리해진다.
- scale selection: 주어진 $r$에 대해, 반지름 $r$인 blob을 인식하기 위해서는 $\sigma = r / \sqrt{2}$로 선택하는 것이 가장 좋다.
- $\sigma^{-2}$에 의존하는 항을 냅두면 $\sigma$가 커질 때 filter가 평평해지는 문제가 있기 때문에 scale normalized LoG를 사용한다.
	- scale normalization: $\nabla_{\text{norm}}^2G = \sigma^2 \nabla^2 G$
### 2-3. Scale Invariant Feature Transform, SIFT
- SIFT는 이미지 회전, 크기에 불변하고, 조명 변화에 강인한 feature point를 찾고 이를 기술하는 것을  목적으로 한다.
- keypoint detector: feature point 찾기
	- finding scale-space extrema
	- keypoint filtering
- keypoint descriptor: feature point를 수학적으로 표현하기
	- orientation assignment
	- calculating descriptors
- interesting patches -> match로 새로운 이미지가 주어졌을 때, 원본 이미지의 어느 부분에서 온 것인지를 추적한다.
#### 2-3-1. Finding Scale-space Extrema
- blob을 추적한다. LoG로도 가능하지만, SIFT에서는 DoG(Difference of Gaussian) 사용한다.
- 서로 다른 bandwidth를 갖는 Gaussian 2개를 빼면 LoG를 효율적으로 근사함을 이용한다. 아래와 같은 근사가 성립한다.
$$\sigma \nabla^2 G = \frac{\partial G}{\partial \sigma} \approx \frac{G(x,y,k\sigma)-G(x,y,\sigma)}{k\sigma - \sigma}$$
- 위의 식을 통해, $G(x,y,k\sigma)-G(x,y,\sigma) \approx (k-1) \nabla_{\text{norm}}^2 G$임을 알 수 있다.
- 이렇게 DoG를 사용하는 이유는 이미지 피라미드에 사용하기 위해서이다. 우선 gaussian을 sigma를 다르게 하면서 여러 개 이미지를 만든다. 그리고, 두 이미지의 차를 계산하면 DoG 값을 얻을 수 있다.
	- 이렇게 하면, kernel의 크기를 작게 해도 된다. 2-dim gaussian filter가 1-dim 2개로 쪼개짐을 명심하자.
	- 더 큰 blob detection을 위해 더 큰 kernel이 필요없다. 그냥 작은 kernel을 유지하면서 그걸 계속 적용하면 되기 떄문이다.
	- 즉, gaussian filter의 decomposition이 연산을 줄인다. gaussian을 많이 적용한 이미지일수록 sigma가 큰 gaussian을 한번 적용한 것과 같고, 이렇게 sigma가 큰 인접한 두 gaussian의 차이는 더 큰 blob을 인식하는 데 사용된다.
- 일정 이상 적용한 다음에는 scale을 줄이고 이를 반복한다. scale 불변성을 만들기 위함이며, aliasing을 방지하기 위해 적당히 중간 정도 gaussian을 적용한 것을 downsampling한다.
- 이렇게 얻은 difference image를 scale을 고려하며 전부 쌓는다. 이를 scale-space라고 하는데, 여기서 $3 \times 3 \times 3$ 크기의 neighborhood에 대해, non-max suppression을 해서 중요한 점만 남긴다.
#### 2-3-2. Keypoint Filtering
- non-max suppression을 해도 사라지지 않는 것들(noninformative)이 있다.
- 그걸 없애기 위해 harris corner detection과 유사한 방법을 쓴다. 그러나, 여기서는 hessian matrix를 활용한다.
- harris는 corner에 더 적합한 반면, hessian은 blob에 더 적합하다. H의 trace가 
$$H = \begin{pmatrix}D_{xx} & D_{xy} \\ D_{xy} & D_{yy} \end{pmatrix}$$
- 여기서 $\frac{(\text{tr} H)^2}{\det H}=\frac{(\gamma + 1)^2}{\gamma}$ for $\gamma = \lambda_1 / \lambda_2$이다.
- $\frac{(\text{tr} H)^2}{\det H} > \delta$인 모든 점을 제거한다.
#### 2-3-3. Orientation Assignment
- 위에서 filtering으로 최종적으로 얻은 각 keypoint에 대해 주 방향을 계산하는 단계이다.
- 우선, 각 keypoint마다 octave, level에 따라 blob의 크기가 결정된다. 이 크기를 결정한 실제 가우시안 필터를 가중치의 형태로 적용한다.
- 이렇게 얻은 데이터의 점별 gradient를 계산해서 edge detection을 한다. gradient 정보를 이용해 아래와 같이 magnitude와 angle을 알 수 있다.
$$m(x,y)=\sqrt{\{I(x+1,y)-I(x-1,y)\}^2+\{I(x,y+1)-I(x,y-1)\}^2}$$
$$\theta(x,y)=\tan ^{-1} \left[\frac{I(x+1,y)-I(x-1,y)}{I(x,y+1)-I(x,y-1)}\right]$$
- 그러면 하나의 block에서 각 픽셀마다 edge detection의 각도를 계산할 수 있다. 그 후, 그것들을 정해진 개수의 방향만 존재하도록 양자화된 공간에서 각 방향으로 projection한다. 그렇게 얻은 방향 중 가장 큰 방향을 이용한다.
- 이것이 dominant orientation을 계산하는 방법이다. 계산이 끝나면 해당 방향을 0으로 옮기도록 회전하고, 이렇게 얻은 keypoint는 이제 scale-invariant이다.
- dominant한 각도가 많으면(최댓값에서 80%까지), 각 방향으로 전부 회전한 복사본을 많이 만든다. 그 중 어느것이 진짜 방향인지 알 수 없기 때문이다.
#### 2-3-4. Calculating Descriptor
- 마지막으로 위의 keypoint에서 dominant하게 만든 각 복사본(blob)을 크기에 따라 회전된 4x4 격자로 자른다.
- 그리고 각 격자마다 다시 orientation을 계산하고 이를 이용해 histogram을 만든다. 이 히스토그램은 (격자 4x4) x (양자화된 8개 방향)으로 총 128차원이다.
- 이렇게 만들어진 128차원 벡터가 해당 keypoint를 나타내는 특징 벡터가 된다.
#### 2-3-5. Summary
- scale invariance: scale-space search
- rotation invariance: dominant orientation estimate
- small variations insensitive: the histogram-based descriptor
- illumination insensitive: the gradient-based descriptor
- noise insensitive: gaussian smoothing before gradient computation
### 2-4. Bonus: Bag of Words
- codewords: independent visual features.
- histogram representation: codeword가 얼마나 빈번하게 나타나는지를 측정한다.
1. codeword construction:
	- interesting point 얻기
	- 격자로 나누기
	- 랜덤하게 찾기
2. 특징 추출: SIFT, SURF 등등
3. codeword dictionary 만들기: 특징 추출기, k-mean clustering 사용.
- limitation: 이미지의 기하학적 특징이 달라도 같게 인식하는 문제가 있음. 이미지의 keypoint들이 이미지에 놓여있는 위치가 중요하지 않음.
#### 2-4-1. Spatial Pyramid Matching
- Bag of Word의 limitation을 해결하기 위해 만든 방법.
- level을 여러 개 만들고, 각 레벨별로 하나의 codeword 얻는 격자 크기 조절한다.
- 그 격자마다 빈도를 찾아서 histogram을 따로따로 만든다.
