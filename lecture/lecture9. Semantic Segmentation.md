### 9-1. Semantic Segmentation
- semantic segmentation: pixel level classification
- segmentation은 이미지 처리시, spatial 차원이 비슷한 pixel을 grouping하는 것이 목적.
- performance는 IoU로 계산. 이제는 pixel level mask로 비교함.
- early approach: selective search가 이미 segmentation을 제공하기 때문에, 그냥 classification만 하면 됨.
	- RCNN을 활용할 수 있음.
	- performance가 region proposal에 의존한다. 근데, selective search는 비지도 이미지 segmentation이기에 성능이 별로 안 좋음.
- 그래서 end-to-end 모델로 만드는 게 그 다음 방법임.
	- 이미지를 받아서, 바로 최종 결과를 내는 확률 분포를 학습시킨다.
	- loss는 groundtruth랑 predicted map이랑 cross entropy loss를 이용해 계산함.
- end-to-end 관련 이슈: resolution of prediction vs size of receptive field
	- receptive field가 클수록 pixel level classification 잘 된다.
	- semantic segmentation에서 각 pixel을 구분하기 위해서는 이게 아주 중요.
	- 이걸 CNN으로 하려면 max pooling을 아주 크게 하거나, CNN 길이를 늘려야 한다.
	- 근데, 이건 메모리가 너무 많이 필요하고, 너무 많이 줄이면 feature map resolution이 작아진다는 문제가 있다.
### 9-2. End-to-End CNN Architectures
#### 9-2-1. FCN
- FCN(Fully Convolutional Network): CNN의 마지막 FCL을 1x1 CNN으로 해석한다.
	- 원래는 하나의 feature vector로 합쳐지고, 그걸 FCL에 넣어서 최종적으로 확률분포를 만들었다.
	- 이제는, 그걸 feature vector 전체에 각각 적용해서 다시 feature vector와 같은 크기의 이미지를 만든다.
	- 그리고 그 결과를 bilinear interpolation을 이용해 다시 원래 이미지 크기로 원복한다.
	- 그러면, 그렇게 만든 결과가 pixelwise classification 결과 이미지임.
	- 이렇게 하면, 임의의 크기와 비율의 이미지를 처리할 수 있음.
	- 단점: 매우 낮은 resolution(pool5 결과)에서 score map을 예측하기에 정확한 위치 정보는 크게 손실됨.
- skip-connection: 정보를 복원하는 과정.
	- upsampling으로 각 pool 과정 직후와 해상도 일치시키고, 그냥 픽셀별로 특징 맵 붙인다.
	- FCN16: pool5의 고수준 의미를 유지하면서 동시에 pool4의 자세한 공간 정보를 복구한다.
	- FCN8: pool4에서 통합한 특징을 다시 2배 업샘플링하고 pool3의 특징을 결합한다.
	- 이런 식의 결합은 채널 차원이 늘어나게 되는데, 이는 마지막에 1x1 conv로 압축해서 전체 채널 수를 예측해야 하는 클래스 개수로 만든다.
	- 1x1 conv는 공간적 해상도는 유지하면서 채널 차원에서 선형 변환을 수행하는 것.
- upsampling은 deconvolution, bilinear interpolation 등의 다양한 방법으로 수행된다.
- 기존의 CNN보다 더 빠르고 정확한 결과를 얻었다.
	- faster: 모델이 end-to-end임. 외부 proposal에 의존하지 않음.
	- more accurate: proposal에 국한되지 않음. feature 표현력과 class 결정이 함께 최적화됨.
#### 9-2-2. DeepLab
- FCN의 개선된 버전
	- atrous convolution
	- fully connected conditional random field(CRF) (post-processing)
	- use ResNet
	- 사실 2, 3번은 이 paper에서 제공된 게 아님. 그래서 진짜 중요한 건 1번임.
- atrous는 프랑스 말로 holes라는 뜻. 구멍이 있다. 중간 중간을 비어있게 만들어 max pooling 없이 receptive field가 점점 커진다. 그 크기를 dilation rate $r$로 정의한다.
- 그런데, 이렇게 구멍이 있어도 되는걸까? 그건 때에 따라 다르다.
	- receptive field는 중복된 정보가 많기 때문에 sparse하게 해도 된다.
	- 당연히 너무 크면 안됨.
	- 그러나, fully-supervised되는 경우에는 큰 문제가 없음.
- CNN의 결과 feature map에 dilation rate를 다르게 한 여러 브랜치(r=6,12,18,24)를 만들어서, 최종적으로 합쳤다. 하나의 앙상블 모델.
- Fully-connected CRF(dense CRF): CRF는 사실 열역학에서 처음 등장함.
	- 원본 이미지에서 한 쌍의 픽셀들이 색상과 위치가 유사하다면, 그들의 score map 값도 비슷해야 한다는 직관에서 등장.
	- score map은 각 class label과 background까지 총 $k+1$개 나온다.
	- 즉, 각 score map에서 두 쌍의 score가 모두 비슷하게 나와야 한다는 뜻임.
	- 이걸 위해서 energy minimization 문제를 푼다. 그냥 하나의 optimization임.
$$
E(x)=\sum_i\phi_i(x_i)+\sum_{ij}\psi_{ij}(x_i,x_j)
$$
- 위 함수를 최소화하는건데, 왼쪽은 unary, 오른쪽은 pairwise potential의 합을 나타낸다.
	- 각 항을 해석한 결과는 다음과 같다.
		- unary term: CNN 최종 score map에서 가져온다. CNN이 예측한 결과가 높으면 이 항이 낮아진다.
		- pairwise term: 이게 중요하다. 두 픽셀을 연관시키는 거랑 관련이 있고, 레이블 일관성을 유지하도록 한다. 2개의 가우시안 커널을 이용하는데, 하나는 위치, 하나는 색상이 가까운 것들을 관리한다.
	- CRF는 optimize가 아주 느리다.
	- 또한, 모델이 이미 잘 예측하는 경우 CRF는 도움 안된다.
	- 다만, 예측한 경계가 너무 둥글때는 도움 된다. DeepLab은 pooling, stride를 사용하기 때문에 최종 분할 맵의 해상도가 낮아서 객체 경계선이 부드럽지 않고 뭉개졌다.
#### 9-2-3. Deconvolution Network
- deconvolution network: 인코더 디코더 구조.
	- 이 네트워크의 개선버전이 u-net이다. 여기서는 안 다룰거임. u-net은 이미지 생성에도 쓰인다.
	- 최종적으로 만들어진 하나의 feature vector를 다시 되돌리기 위해, unpooling과 deconvolution layer가 있다.
	- unpooling은 pooling한 위치 그대로 되돌려놓는다. 나머지 위치에는 0을 집어넣기 때문에, 이렇게 나온 결과는 sparse하다.
		- max pooling이라면, max를 택한 위치를 기억하는 거임. (switch variable)
	- 그렇게 sparse한 데이터를 다시 채워주는 게 deconvolution이다.
	- 즉, deconvolution은 learnable densification이다.
- 구조를 복원하고, end-to-end trainable한게 장점임.
### 9-3. Instance Segmentation with CNNs
- segmentation은 pixelwise classification을 한다. 따라서, 같은 class의 다른 물체는 구분 못한다. 즉, 사람 여러명을 하나의 하나의 클래스로 구분한다.
- instance segmentation: 각 클래스 내의 각 물체를 별개로 탐지.
- performance metric: IoU criterion을 특정 thres로 잘라서 만든 precision, recall을 이용한 AP(average precision) 계산.
- early approach:
	- SDS(Simultaneous Detection and Segmentation): 하나의 프레임워크.
		- segmentation의 초기 단계 구조.
		- box를 위한 특징 표현과 region을 위한 특징 표현을 따로 학습시키고 둘을 결합해 최종 예측에 사용한다.
	- MCG(Multiscale Combinatorial Grouping): SDS의 2번째 단계인 proposal을 제안하는 알고리즘.
		- 이미지 피라미드를 만들고, 각각에서 독립적으로 segmentation을 수행한다.
		- 그 후, 이 다중 스케일 구조를 정렬하고 결합해서 단일 통합 계층을 만든다.
		- 신경망이 MCG가 제안한 후보 영역만 집중하게 하기 때문에 성능이 좋다.
	- CFM(convolutional feature masking): SDS 단점 개선하는 프레임워크
		- Fast RCNN의 아이디어를 이용함. 달리 말해, SDS는 각 영역마다 CNN을 했는데, CFM은 전체 이미지에 대해 한번만 한다.
		- 그게 특징 맵에서 해당하는 위치를 계산하고 특징 벡터를 뽑아낸다.
	- 단점: segmentation이 region-proposal에 의존한다.
- 해결책: end-to-end CNN
	- 더 이상 주어진 region에 의존하는게 아니라, 그냥 3가지 작업을 동시에 한다.
	- object localization, mask prediction, object classification
#### 9-3-1. Multi-task Network Cascades
- detection, segmentation, classification을 순차적으로 수행하는 하나의 프레임워크이다.
	- conv feature map을 우선 만든다.
	1. RoI를 찾아낸다.
	2. RoI와 conv feature map을 이용해 RoI warping/pooling으로 영역 추출하고, 그걸 flatten해서 FCL에 넣고, 그 결과를 logistic regression을 이용해 최종 mask instance를 만든다.
		- 입력은 채널을 고려하면 몇 만 차원인데, 출력은 1채널이 된다.
	3. mask instance를 feature map에 적용해서 자른다. 그걸 FCL에 넣어서 최종 class 예측한다.
- cascade라는 이름은 이전 단계의 출력이 다음 단계로 입력되어 성능을 향상시키는데 쓰이기 때문이다.
	- 각 단계에서 이전 단계의 정보가 다음 단계에 도움이 된다.
	- stage 1의 RoI는 stage 2의 mask 생성에 도움을 준다.
	- stage 2의 mask는 stage 3의 분류를 정확하게 만드는데 쓰인다.
- 한계: FCL은 output을 $n^2$ 벡터로 든다. 즉, FCL의 연산 크기가 크기 때문에 mask resolution을 키울 수가 없다(당시에는 28x28을 이용함). 이에, 예측 결과가 둥글게 만들어진다.
#### 9-3-2. Mask R-CNN
- RCNN 계열 모델. 간단하지만 강한 모델. Faster RCNN을 instance segmentaiton으로 확장한거임.
	- RoI align: RoI pooling을 대체하는 기법. cell-wise하게 linear interpolation으로 quantization 문제를 해결함. 이 방식은 겹쳐진 픽셀도 잘 반영된다.
	- fully conv segmentation branch: 각 RoI mask를 예측하기 위한 브랜치. MNC는 FC를 썼던 것과 달리 그냥 fully conv net을 쓴다.
		- RoI를 입력받아서 deconv로 업샘플링해서 높은 해상도의 마스크를 출력한다.
		- 여기서 deconv는 unpooling이 들어가는 deconv net에서의 개념이 아님. 필터가 겹치면서 연산하여 입력 값을 여러 출력 위치로 분산시키고 보간하는 그냥 하나의 레이어임.
- deconv에 대한 설명
- $x=(x_1,x_2,x_3,x_4)$가 있다고 하자. kernel $k=(k_1,k_2,k_3)$와 stride 2가 있다고 하면, 그 입력은 다음과 같다.
$$
C=
\begin{pmatrix}
k_1&k_2&k_3&0\\
0&0&k_1&k_2\\
\end{pmatrix}
$$
- 즉, $Cx$가 conv를 적용한 결과가 된다.
- 반면, deconv는 $y=(y_1,y_2)$에 대해, $C^Ty$를 수행하는 것이다. 이는 $y_i$가 각 픽셀에 기여하는 가중치가 어느정도인지를 계산한다.
- 이러한 이유 때문에, deconv는 transpose conv라고도 불린다.
#### 9-3-3. Leverage Context
- ==이 파트는 별로 안 중요함==
- Pyramid Scene Parsing Network
- Context Encoding Network
	- context: 전체 이미지에서 얻을 수 있는 semantic한 정보
#### 9-3-4. Feature Upsampling
- ==이 파트는 별로 안 중요함==
- CARAFE(Content-Aware ReAssembly of FEatures)
- Dual Super-resolution Learning
### 9-4. Semantic Segmentation with Transformers
- segmenter: 그냥 transformer임.
	- 인코더는 ViT에서 가져옴. 그때는 목적이 image level classification이었음.
	- 디코더는 encoder에서 가져온 벡터와 class token(learnable embedding임)을 함께 넣는다. 각 learnable query가 classifier의 역할을 수행한다.
	- 디코더에 의해 이미지 임베딩과 class token이 섞이게 되고(클래스 쿼리가 이미지 패치 중에서 자신이 대표하는 클래스랑 관련된 피처에만 주의를 기울여(attend) 추출하고 정제해서 마스크 임베딩으로 변환한다.), 그걸 각 classification 벡터랑 스칼라곱해서 임베딩별로 class 개수의 마스크를 얻는다.
	- 마지막으로, 이걸 upsampling + 픽셀 위치마다 mask 중에서 max인걸 골라서 최종적으로 segmentation한다.
		- upsample은 그냥 bilinear interpolation으로 수행하는데, 어차피 ViT랑 mask transformer가 강력해서 복잡한 업샘플링이 필요가 없음.
- decoder는 mask transformer. 그런데, metric을 보면 linear decoder랑 크게 다르지 않다는 걸 알 수 있음.
