### 8-1. Object Detection
- 목적: pre-defined class에 대한 정확한 bound box 그리기.
- detection: localization + classification
- localization: sliding windows, object proposal, branch-and-bound serach 등의 search space를 줄이는 방법들
- classification: classifier가 class 분류.
	- classifier 존재하면, sliding window로 모든 이미지를 훑으면서 찾고자 하는 것의 위치를 탐색하면 됨.
	- window의 크기도 바꿔가면서 해야 됨.
- 앞선 강의에서 이미지를 numerical form으로 바꾸는 것, classification function을 찾는 건 했음.
- 이제 남은 건, 이미지에서 local object 후보를 가져오는 것(localization). 이걸 잘하는게 목표임.
- object proposal: sliding window보다 더 적은 양의 후보를 만드는 거. 그러면 속도가 빠르다.
	- selective search algorithm: 간단한 image segmentation을 한다. 이건 신경망 아님. 그리고 그걸 점점 더 합쳐가면서 큰 segment를 만든다. 작고 큰 image segment에 대해 전부 bounding box를 그린다.
	- 보통 sliding window는 가로세로 비율이 고정되어 있는데, selective search는 그럴 필요가 없음.
#### 8-1-1. Performance Metric
 - intersection over union(IoU)을 사용. 실제 bound랑 예측한 bound가 겹치는 영역의 넓이를 실제 bound 넓이로 나눠서 구함.
- TP(True Positive): 물체 존재 & 그 물체의 bound를 잘 예측
- FP(False Positive):
	- 물체 존재 & 그 물체의 bound를 예측. 그러나 잘 예측 못함.
	- 물체 없음 & 근데 있다고 예측함.
- FN(False Negative): 물체 존재 & 그 물체를 예측 못함.
	- Negative: 모델이 bound box를 주지 않았다는 말.
- precision = (# of TP)/(# of Predictions). 1이 최댓값.
- recall = (# of TP)/(# of GT). GT: ground-truth. truth를 얼마나 잘 예측하는지.
- threshold를 정하고, 그 이상의 confidence를 가진 bound 예측만 남긴다.
	- threshold가 작으면, confidence가 낮은 예측도 평가에 포함되기 때문에, precision은 내려가고, recall은 올라간다.
	- 반대로 threshold가 크면, precision은 올라가고, recall은 내려간다.
- average precision: threshold를 바꿔가며 Pre, Rec를 각각 x, y축에 그리면, 위로 볼록한 감소 그래프를 그 수 있다. 그 전체 평균을 계산해서 최종적으로 모델의 성능을 evaluation한다.
	- 당연히 practice에서는 threshold를 너무 dense하게 설정하지는 않고, 적절히 설정한다.
### 8-2. Pedestrian Detection with SVMs
#### 8-2-1. Histogram of Oriented Gradients
- 기본 아이디어는 SIFT랑 비슷함.
- 다만, gaussian smoothing을 쓰지 않고, 간단한 1x3차원 벡터로 가로 세로 선을 계산함. (SIFT랑 다른 부분)
- 가로 세로 선 정보를 모아 만든 2dim 벡터를 통해, 실제 orientation을 계산한다. 그걸 다시 8개의 quantized orientation 중 최대 2개로 project된다.
- magnitude가 너무 작으면 무시한다. 각 magnitude마다 soft voting을 해서 히스토그램을 만든다.
- 색 정보가 있으면 각 채널 중에서 고르고 최대 그레디언트를 선택한다.
- cell: 8x8 픽셀 영역. orientation histogram 만드는 단위 영역
- block: 2x2 cell.
	- 다음 영역으로 이동 시에 cell을 공유하며 부분적으로 겹칠 수 있음.
	- 4개의 히스토그램을 하나로 연결(concatenate)해서 초기 블록 특징 벡터를 만든다.
	- 연결한 히스토그램을 정규화해서 조명이나 그림자 변화 등에 대한 robustness를 확보한다. (L-2 norm 사용)
- window: 8x16 cell: 내부의 block descriptor를 합친다(concatenate). 객체를 최종적으로 추출함.
#### 8-2-2. Sampling Train/Eval Data
- unbalance between positive, negative data.
- 당연히 negative data가 훨씬 더 많음.
- 거기서 모델의 성능을 올릴 hard negative를 찾는게 중요함.
- 개수를 맞추는게 모델 성능에 더 좋음.
- linear SVM을 soft margin과 함꼐 사용한다.
### 8-3. Object Detection with CNNs
- object detection = box localization + box classification
- 우리가 했던 건 classification 모델을 CNN으로 바꾼 거임.
- 이제는 localization도 CNN이 하게 한다.
#### 8-3-1. R-CNN
- Region-based CNN
- 데이터를 받고, 그걸 CNN으로 넣는다. 입력값은 정해진 비율이 되도록 crop, resize한다.
- meta-framework이고, proposal 방법이나 CNN 아키텍처, classification 방법은 아무거나 넣어도 됨.
- bounding box regression(transformation): ==매우 중요==
	- 기존에는 selective search를 이용해 object proposal을 했다.
	- 이런 object proposal algorithm 들은 pixel level clustering을 쓴다. (unsupervised)
	- 이런 알고리즘들은 localization 성능이 별로 안 높다.
	- 아이디어: object proposal 하고, 그걸 ground-truth box로 위치변환을 하는 bounding box transformation + scaling을 따로 학습시키는 NN을 만든다. → 총 4개의 파라미터 필요.
	- 가장 naive한 방법은 parameter 4개를 그냥 학습시키는 거다.
		- 이건 별로 안 좋음. 왜냐면, RCNN은 CNN에 넣는 이미지를 고정된 크기로 만들기 때문이다.
		- 즉, CNN 레이어는 항상 고정된 크기의 이미지를 받기에 원본 bound box의 scale을 모름.
		- transformation을 original scale을 데이터가 없는 상태로 학습시키면 그건 one-to-many 함수이다. 이런 함수는 학습이 불가능하다.
	- 여기서 $d_x(P_w)$, $d_y(P_h)$는 $P_w$, $P_h$에 대한 상대량이다.
		- 실제 결과는 predict한 변위에 기존 데이터 scale을 곱해 $P_w \cdot d_x(P)$를 얻고, 그걸 원본 값에 더한다.
		- width, height에 대한 scale은 항상 양수여야 하기에 $d_w$, $d_h$에는 exp를 씌운다.
		- 학습에는 CNN의 pool5를 통과한 feature를 사용한다.
		- $d_\alpha(P) \approx w_\alpha^T \phi_5(P)$이다. 여기서 $\phi_5$가 pool5까지 통과한 feature임.
	- 결과적으로, CNN은 그냥 오차가 포함된 형태로 예측을 끝마치는데, 그걸 새로운 bounding box regression 모델이 보정해주는 것이다. 즉, 이건 CNN이랑 별도로 학습되고 사용됨. 
	- 데이터 $d_\alpha^{(k)}$도 상대적인 양으로 별도로 구성한다. 
#### 8-3-2. Fast R-CNN
- RCNN의 빠른 버전.
	- RCNN의 computation bottlenect은 CNN에 각 bound box를 넣어줘야 한다는 거다.
	- 보통 selective search는 bound box를 2000개 정도 만들기에 속도가 엄청 느리다.
- 해결책은 전체 이미지를 CNN으로 딱 1번만 돌리고, feature map을 구한 다음에, 모든 처리를 이 feature map에서 수행하는 것이다.
	- 이때, selective search에서 제안한 RoI를 그대로 conv5 feature map에 투영(feature map은 원본에 비해 크기가 작음)하는 RoI projection 단계가 있어야 한다.
	- 그리고, 이 각 영역에서 고정된 크기(7x7)의 특징 벡터를 추출(conv5는 channel이 여러개이기 때문에 최종 pool된 정보도 벡터 49개임)하는 RoI pooling을 만든다.
	- RoI pooling: 투영된 위치를 quantize(양자화, 정수로 만들기)하고, cell-wise pooling해서 최종적으로 7x7 feature map을 만든다. cell이 7x7 영역에 되도록 하는거임.
	- RoI align: 더 개선된 RoI pooling.
		- misalignment: quantize하면 정보가 날라감. RoI align은 그걸 interpolate해서 해결함.
- 마지막으로 FCL를 넣어서 최종 특징 벡터를 추출하고, 그걸 다시 FC에 넣어서 softmax로 classification한다.
- 같은 최종 특징 벡터를 FC에 넣어서 regression도 한다. 이건 localization 개선을 위해 있는거임.
- 일단 RCNN은 SVM과 linear regression이 분리되어 있었는데, 여기서는 하나로 합쳐져있다.
#### 8-3-3. Faster R-CNN
 - Fast RCNN보다도 빠른 버전이다.
	 - Fast RCNN의 bottleneck은 object proposal에 의존한다는 점.
	 - object proposal은 결국 GPU 밖에서 별도로 수행되기에 속도가 느리다.
- Faster RCNN은 region proposal network(RPN)를 만든다. ==이거 더 이해하기==
- Fast RCNN처럼 CNN 돌려서 feature map 만든 다음에, 그걸 RPN에 넣어서 proposal을 예측한다.
	- sliding window 방식(보통 3x3)으로 스캔한다. 잘라진 각 영역이 그것의 receptive field를 대표한다.
	- 그 영역을 256차원의 중간 계층을 거치고, cls layer랑 reg layer로 옮긴다.
	- 각 슬라이딩 윈도우마다 k개의 anchor box를 정의한다. 이러한 후보군은 가로세로 비율, scale을 다르게 다르게 해서 여러 개 만든다.
		- cls layer: $2k$ score. 각 anchor box가 객체일 확률과 배경일 확률을 총 $k$개 쌍으로 출력
		- reg layer: $4k$ coords. 각 anchor box가 정답 박스를 가깝게 조정하기 위한 4개의 변환을 $k$개 출력
	- cls layer에서 높은 점수를 받은 anchor를 선택하고, 그거의 reg layer 변환 값으로 보정해서 최종 proposal을 만든다.
	- 이렇게 만들어진 proposal은 다시 feature map으로 전달되어, RoI pooling으로 고정된 크기의 특징 벡터를 추출한다.
	- 마지막으로, 이걸 classifier랑 bbox regression을 통해 최종 결과를 출력한다.
	- regression이 상당히 어려운 작업이기 때문에 바로 box를 예측하는 것이 아니라 후보군을 만들어두고 거기서 선택하는 것이다.
- backbone 모델이 좋으면 성능도 좋아진다. CNN 위치에 ResNet쓰면 성능이 상당히 좋다.
==8-3-4부터 8-3-6은 현재는 생략==
#### 8-3-4. Feature Pyramid
- 목적: 스케일이 다른 객체를 인식하기 위함.
- feature pyramid network(FPN): upsample feature map.
	- 제한된 resolution에서 limited setups을 핸들링한다.
#### 8-3-5. Rethinking the Two Heads
- double headed detector
#### 8-3-6. Sample Weighting Strategies
- sample weighting network(SWN)
- prime sample attention(PISA)
### 8-4. Object Detection with Transformers
- 일단 ViT 나오기 전에 만들어진 거임.
- 전체가 transformer인 건 아니고 초반에 CNN 있음. 그러나, key concept가 중요함.
	- 우선, backbone의 CNN으로 feature map을 추출하고, 거기에 positional encoding한다.
	- 그걸 transformer encoder에 넣는다. 각 feature vector가 서로에게 영향력을 행사할 수 있기 때문이다.
		- 이렇게 transformer를 이용하면 예측하려는 receptive field의 크기가 동적으로 변할 수 있다.
	- encoder의 토큰 임베딩이 decoder로 간다.
	- decoder는 초기 값으로 object query를 받는다. 이것들도 학습가능한 parameters다.
	- decoder의 최종결과로 나온 벡터들이 FFN(3 layer MLP with ReLU)에 들어가서 `[class, box]`를 반환하거나, `no object`를 반환한다.
	- object query 개수만큼 object를 찾을 수 있음.
- loss function: object query로 얻은 bound box를 비교한다(IoU).
	- 미리 정의된 anchor나 proposal을 IoU로 연결하는게 아니라, 최소 비용 매칭을 사용한다.
	- 예측 집합과 정답 집합이 가장 비용이 적게 들도록 연결하는 순열을 탐색한다.
	- 각 쌍의 매칭 비용은 클래스 예측 비용과 바운딩 박스 예측 비용의 합산이다.
	- 이렇게 찾은 최적 쌍에 대해서 Loss를 계산한다. 이 최종 손실을 최소화하는게 목적이다.
	- 단점
		- matching은 미분 안 됨. 논문에서는 헝가리안 알고리즘을 쓰는데, 계산 부하가 큼.
		- epoch마다 matching이 달라질 수 있다. 달라지면 다시 계산해야 됨.
		- unstable training을 유발시키는 경우가 있음. (예측이 미세하기 바뀌어도 최적 순열이 달라짐)
- non-maximum suppression이 필요가 없다.
	- 어차피 계산할 때, 최소 비용 매칭을 강제하기 때문.
	- 이미지에 객체가 3개면, N개 예측 중 3개만 실제 객체에 매칭되도록 손실을 계산한다.
	- 즉, 중복된 객체를 예측하거나, 다른 객체를 예측하면 그만큼 손실이 커진다.
	- 이게 알아서 이중 prediction을 penalize한다.
