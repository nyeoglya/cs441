### 11-1. 3D Convolution
- 2D conv의 확장판이다. 2D는 spatial 차원만 훑는 반면, 3D는 temporal 차원도 훑고 지나간다는 특징이 있음.
- 비디오는 모션 정보를 통해 더 잘 이해할 수 있다. 이에, spatial, temporal 상에서 3D conv를 이용해 동시에 정보를 추출하는게 목적이다.
- 또한, 비디오는 많은 이미지로 구성되어 있기 때문에 이를 효율적으로 처리할 수 있어야 한다.
- 첫번째 formulation은 ICML 2010에서 나옴.
	- 입력값은 60x40 grayscale, 7 frame 비디오.
	- hardwired: 우선 채널을 선천적으로 5배($grayscale, g_x, g_y, opt_x, opt_y$)로 늘린다.
		- 왜 manual하게 채널을 늘리는 단계가 있을까?
		- 옛날 paper라서, 큰 데이터셋이 없었고, gpu도 안 좋았기에 큰 규모의 NN 만드는 거 어려움.
		- 근데, gradient랑 opt flow가 도움이 되는 정보인 건 분명하기 때문에 이렇게 구성.
		- opt flow는 마지막 프레임에서 불가능하기에 hardwired 단계는 33개가 만들어진다.
	- activation은 tanh를 썼다. 각 채널을 전부 가중치와 함께 더하고 tanh를 사용하는 방식.
	- 결론적으로 마지막에 긴 벡터 하나가 나와서 이걸 FCL에 넣고 학습을 시킨다.
	- 일단 성능이 별로 안좋음. shallow하고 영상 데이터도 적었음.
	- motion estimation을 명시적으로 수행했기에 3d conv의 이점을 전부 살리지 못함.
	- 심지어 hand-crafted 보다도 안 좋다.
- C3D: 첫번째 modern CNN 아키택처 (i.e. 3D VGGNet)
	- ICCV 2015에서 나옴. 이때는 비디오 데이터가 많아서 이 모델이 SOTA를 달성할 수 있었다.
	- 8개 conv와 5개 pool로 구성되어 있다.
		- resolution이 3x3x3일때 가장 성능이 좋았음.
		- 2x2x2 max pool with stride 1
		- 마지막에 FCL 2개
	- C3D는 hand crafted feature(iDT, improved dense trajectory)를 못 넘었다.
	- 대신, gpu에서만 작동가능한 구조이기 때문에 속도가 iDT보다 훨씬 빠르다. 당연히 비용이 여전히 많이 들긴 함.
	- 일단 temporal receptive field가 제한적임. resolution이 3밖에 안되기 때문에, 넓은 시간 범위의 모션을 인식하는 건 안됨. (즉, 큰 모션을 인식하는건 어려움)
### 11-2. 2D Feature Fusion Across Time
- 비디오 = 정해진 크기의 clip의 집합. 각 clip을 classification하고, 그것들을 합쳐서 최종 결정에 쓴다.
	- 이미지 여러 개를 섞은 정보를 다루는 방법을 알아야 한다.
- 4가지 정보 fusion 방법
	1. single frame: 프레임 하나만 보기. 각 프레임을 이미지 수준으로 분석.
	2. early fusion: conv kernel이 spatial dim만 돈다. 특정한 시간 수준을 전부 mix한다. temporal dim은 상당히 길게 잘라서 motion 정보를 볼 수 있다.
	3. late fusion: 일종의 앙상블 기법. 15칸 떨어진 이미지 2개를 골라서 같은 CNN에 넣고, 그걸 합친 다음에 FCL에 넣는다. 첫번째 FCL이 두 이미지의 정보를 섞는다.
	4. slow fusion: early + late. 연속된 이미지 4개씩 자른다. 그리고, 그것들을 CNN에 넣고, 진행하면서 조금씩 합친다.
- 근데 사실 위의 fusion은 별로 도움이 안된다.
- multi-resolution CNN: 비디오 인식은 계산이 매우 오래 걸린다. 그래서, spatial resolution(해상도)를 낮춘다. 당연히 정확도는 좀 떨어진다.
	- fovea stream: 사진의 중앙만 도려낸다. 중요한 정보는 보통 중앙에 있다.
	- context stream: 다운샘플링한다.
- 이런 fusion이나 multi-resolution이 필요한가에 대해서는, 데이터에 따라 다르다.
- 그렇지만 알고봤더니 이 사람들이 학습한 정보가 모션 정보가 아예 필요없는 것이라, 그냥 해도 잘 나옴.
- 즉, 위의 아키텍처 디자인은 사실상 틀린거임.
### 11-3. Computing and Using Motion Explicitly
- Two Stream Convolutional Network: NeuraIPS 2014에서 소개함.
	- 사람의 visual cortex에서 영감을 받았음.
		- ventral stream: 물체 인식. (=spatial stream)
		- dorsal stream: 모션 인식. (=temporal stream)
	- 2개의 stream 사용한다.
		- spatial stream: 정적인 이미지에서의 action recognition. 프레임마다 rgb 데이터 받음.
			- ZFNet을 사용. 이거 자체만으로도 굉장히 잘하는 놈임.
			- 두 stream을 분리했기 때문에, spatial stream에는 ImageNet 같은 큰 이미지 데이터들로 별도의 학습을 할 수가 있다는 장점이 있음.
		- temporal stream: dense motion field에서 action recognition. optical flow 받음.
			- ZFNet과 유사한 아키텍처. optical flow를 합치고 그걸 받는다.
			- 이때는 이런 optical flow 데이터셋이 적어서 처음부터 학습하기가 어려웠음.
			- 해결책: multitask learning. 네트워크를 동시에 두 개의 다른 비디오 분류 작업에 대해 학습시킨다. 최종 분류를 위한 독립된 헤드를 각각의 데이터셋에 맞게 추가한다. 이러면, 각 데이터셋의 특징을 상호보완적으로 학습하여 네트워크 일반화 능력이 올라간다. (반대로 과적합 위험은 작아짐)
	- 마지막으로, 각 stream에 의해 softmax로 얻은 class score를 합쳐서 최종 결정을 내린다.
	- 성능이 hand-crafted인 iDT보다 더 좋아지긴 했는데, 크게 좋아지진 않긴 함.
	- 단점은 파라미터가 2배이고, optical flow가 너무 계산하기 힘듦.
		- 가능한 해결책은 feature space에서 motion 정보를 계산하기.
- MotionSqueeze: pixel level 대신에 feature level motion info를 구한다.
	- 인접한 feature map에서 motion flow를 계산한다.
	- correlation computation: feature map에서 vector 가져오고, $(2k+1)^2$개 인접한 사각형 패치와 계산해서 최댓값 찾으면, 어디로 이동한건지 알 수 있다.
	- 모션 방향을 찾기 위해 각 변위랑 그에 따른 가중치를 곱하는데, 그냥 곱하면 데이터가 너무 블러되기 때문에 gaussian kernel 이용해서 이러한 블러를 방지한다.
	- 마지막에는 이렇게 얻은 u, v 변위 2개와 채널 축으로 max pool해서 얻은 최대 상관 값(픽셀 움직임 추정을 얼마나 정확하고 신뢰할 수 있는지를 나타내는 값)까지 해서 3개를 separable conv에 넣어서 최종적으로 입력값이랑 같은 차원으로 복원한다.
- 비디오 처리는 temporal 차원 때문에 아직도 인공지능이 잘 수행하지 못하는 분야이다.
