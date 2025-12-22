### 6-1. Introduction to NN
- 뉴런을 모사한 것. 주어진 input $x=[x_1,x_2,\cdots,x_d]^\intercal$이 주어지면, output $y \in \mathbb{R}$을 반환한다.
$$
y=f(z)=f\left(\sum_i w_ix_i + b \right) = f(w^\intercal x + b)
$$
- 여기서 f는 activation function이다. 비선형성을 제공한다. w는 가중치, b는 편향이다.
#### 6-1-1. Training Perceptron
- loss 함수를 최소화해야 한다. 그러기 위해 경사하강법을 쓴다.
- back propagation으로 가중치를 업데이트하는데, 여기에 사슬 규칙을 쓴다.
- 1 epoch: 모든 데이터에 대해 가중치 업데이트 한번 하는 거.
	- 여러 epoch를 지나며 local minimum을 찾는다.
### 6-2. MLP
- layer를 쌓는다.
- hidden NN이 있다.
- 비선형적인 데이터를 다룰 수 있다.
#### 6-2-1. Loss Functions
- 확률분포 $p$가 주어졌을 때의 entropy는 다음과 같다.
$$
H(p)=E_p[-\log p] = -\sum_x p(x)\log p(x)
$$
- cross entropy는 다음과 같다. 여기서 $p$는 ground-truth label 분포이고, $q$는 예측한 label 분포이다.
$$
H(p,q)=E_p[-\log q] = -\sum_x p(x)\log q(x)
$$
- cross entropy loss는 다음과 같다.
$$
L(y,i) = -\log (\exp(y_i)/\sum \exp(y_j))=-y_i+\log(\sum \exp (y_j))
$$
	- 활성화 함수와 손실 함수의 결합 형태.
	- 각 logit을 지수 함수로 변환한다. 로짓을 확률 분포로 바꾼다. (정규화한다는 말임)
	- 확률 분포이기 때문에 앞부분 항이 없어진거임.
- multilabel soft-margin loss: cross entropy loss를 확장해서 이용한다.
- mean square error: regression에 주로 사용됨.
#### 6-2-2. Stochastic Gradient Descent (SGD)
- 만약 샘플이 잘못 label되면, 그거에 영향받아서 문제가 생기기에 때문에 minibatch SGD를 만든다.
- batch 크기는 보통 GPU 램 크기로 정한다.
- 성능적으로도 더 빠르다.
#### 6-2-3. Momentum
- 학습할 때 이전의 방향을 일부 참고하게 한다.
- oscillation이 줄어들고, 수렴이 더 빨라진다.
- 아래와 같은 수식으로 정의된다. 직전 기울기 값을 통해 계산하는 것을 확인할 수 있다.
$$
v_i(t)=\alpha v_i(t-1)-\epsilon \frac{\partial E(t)}{\partial w_i}
$$
- $w_{t+1}=w_t+v(t)$로 업데이트한다.
#### 6-2-4. Issues in Learning MLPS
- overfitting: 학습 데이터랑 너무 비슷한 함수를 만든다.
	- 데이터를 더 많이 써서 해결
	- dropout 같은 규제를 거는 방법도 있다.
- large amount of training time: CPU 써서 계산하는 것이 너무 느리다.
	- 그냥 GPU 쓰면 됨.
- vanishing gradient problem: sigmoid 양 끝 부분이 기울기가 0에 가까워, 학습이 거의 안 이루어진다.
	- ReLU 써서 해결
### 6-3. Convolutional NN
- MLP는 이미지와 비디오에 적합하지 않다. 이미지는 사이즈가 커서 모델이 overfit되기 쉽다.
- convolutional neuron: kernel이 input 크기보다 작은 경우
	- sliding window이다.
	- matrix를 반환한다.
- 커널 크기가 input 크기랑 똑같으면 그건 그냥 perceptron이다. perceptron은 값을 반환한다.
- 즉, perceptron의 일반화된 것이 conv. neuron이다.
#### 6-3-1. CNN
- convolutional layers + pooling + MLP로 구성되어 있다.
- convolutional layers: convolutional kernel로 연산.
	- input 데이터의 channel 크기와 kernel의 channel 크기는 같아야 한다.
	- 만약, kernel이 1개가 있다면, channel 크기가 1인 feature map이 나온다.
	- kernel을 64개를 만들면, channel 크기가 64인 feature map이 된다.
	- 이렇게 만든 feature map을 element-wise하게 ReLU에 넣는다.
	- 데이터가 축소되는 것을 방지하고 싶다면, 0-padding을 한다.
- receptive field: feature vector(가령, channel 크기가 64면 64차원 벡터임)를 만들기 위해 필요한 영역.
	- kernel을 한번씩 적용하면 점점 데이터가 압축되면서 vector가 생긴다.
	- 그 vector를 만들기 위해 역으로 필요한 영역을 의미한다.
- pooling: 데이터를 줄이는 과정.
	- window + stride(window를 움직이는 정도)를 정한다.
	- pooling도 종류가 많음.
	- 메모리 사용을 줄일 수 있다.
	- pooling을 하면 같은 크기의 kernel이 더 넓은 receptive field를 다룰 수 있어, 보다 거시적인 이미지 특징을 얻을 수 있다.
- 결과적으로 channel의 크기는 점점 늘려가며 resolution은 줄여간다.
- 최종 activation map은 하나의 vector로 치환된다.
	- global pooling 아니면 concatenation을 이용.
- 앞쪽 레이어는 edge나 blob 같이 저수준의 특징을 잡는 반면, 뒤쪽 레이어는 텍스처나 모양 같이 보다 추상적인 특징을 잡는다.
### 6-4. Techniques to Avoid Overfitting
#### 6-4-1. Dropout
- 학습 과정에서 Fully Connected Layer의 일정 비율을 비활성화시킨다.
- 학습해야 하는 파라미터들이 줄어든다.
- 앙상블 모델의 효과(여러 개 모델의 결과를 조합해서 하나의 결과를 내는 방법)를 준다.
	- 각 학습 단계에서 일부 뉴런을 무작위로 비활성화하며, iteration마다 뉴런의 비활성화되는 조합이 달라진다.
	- 이는 가중치를 공유하는 수많은 sub-network를 동시에 학습시키는 효과를 준다.
- 또한, 특정 뉴런의 데이터에 대한 과도한 의존성을 막는다.
- 보통 CNN에서는 뒤쪽의 MLP에 dropout을 적용함.
#### 6-4-2. Weight Decay
- error fucntion에 weight decay을 추가함으로써, degree of freedom을 줄이고 overfitting 막는다.
- error function을 아래와 같이 바꾼다.
$$
\tilde{E}=E+\frac{\lambda}{2}\sum_i w_i^2
$$
- weight update rule을 아래와 같이 바꾼다.
$$
w_i \leftarrow w_i - \epsilon \frac{\partial E}{\partial w_i} - \lambda w_i
$$
#### 6-4-3. Early Stopping
- 데이터를 training, validation으로 분리한다.
- validation 데이터들은 evaluation 단계에서만 사용한다. 그러면, 특정 지점에서 멈춤으로서 overfit을 방지할 수 있다.
### 6-5. Weight Initialization
- learn from scratch: 요즘은 쉽지 않음. 그냥 좋은 foundation model을 받아서 사용하면 됨.
	- 얕은 network에는 그냥 gaussian을 따르는 숫자로 초기화하면 됨.
	- 근데 이렇게 하면 deep network에서는 layer가 지남에 따라 activation이 0이 됨.
	- xavier initialization: zero mean + variance $\frac{2}{n_\text{in}+n_\text{out}}$
	- he initialization: zero mean + variance $2/n_\text{in}$
- finetuning an existing network: 이건 됨. 내 데이터에 적합하게 잘 학습된 모델을 조정한다.
- 초기화는 라이브러리가 다 해줌.
### 6-6. Optimization
- 원래 learning rate을 manual하게 조정했다.
- SGD를 쓰면 learning rate가 상당히 중요하다.
- 요즘은  adaptive하게 learning rate가 파라미터 단위로 조절된다.
- 알고리즘이 많은데 뭐가 좋은지는 해봐야 안다. adam 같은 것들이 있음.
