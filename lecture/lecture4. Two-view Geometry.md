- Note: 3D vision에 대한 유일한 단원이다.
### 4-1. Two-view Geometry
- 두 이미지가 주어졌을 때,
	- 첫번째와 두번째 이미지에서 대응되는 점이 어떻게 되는가?
	- 카메라로부터의 상대적 거리가 어떻게 되는가?
	- 3D geometry가 어떻게 되는가?
- application: depth estimation, 3d reconstruction
### 4-2. Homography
- 직선은 변환 이후에도 직선을 유지한다. (=collinearity가 보존됨)
	- 선을 구성하는 점 3개 옮겨서 확인해보면 됨.
- homography로 같은 3d 세상에서 2개의 서로 다른 2d 이미지의 관계를 분석할 수 있음.
	- 하나의 이미지에서 다른 이미지로 옮기는 homography를 찾으면 된다.
#### 4-2-1. Homography Estimation
- 두 평면 이미지에서 알고 있는 대응 관계에서 시작해 homography matrix를 구해야 한다.
- 우리가 correspond하는 두 점 $(x_i,y_i)$와 $(x'_i,y'_i)$를 알고 있다고 하자. 그러면 아래와 같은 관계가 성립한다.
$$\begin{pmatrix}x_i'\\y_i'\\1\end{pmatrix} \propto H  \begin{pmatrix}x_i\\y_i\\1\end{pmatrix}$$
- 하나의 대응 관계에 대해 2개의 방정식이 주어질 것이고, $H$는 DoF가 8이기 때문에, 적어도 4개의 대응 관계를 알고 있어야 한다.
	- 이때, 그러한 4개의 대응관계 중 어느 3개도 collinear한 관계면 안된다. 만약 그러면, 그 중 하나는 linear combination으로 인해 겹치기 때문이다.
	- 오차를 줄이기 위해, total least square method를 사용한다.
	- 행렬 내부의 값 차이가 크면 eigen decomposition이 불안정해진다.
	- 데이터를 원점으로부터 평균거리가 $\sqrt{2}$가 되도록 정규화해서 해결할 수 있음.
	- 정규화 행렬 $T$에 대해, 변환된 정보로 homography $H$를 얻었으면, 원본 좌표에서 계산할 때는 $T^{-1}HT$를 적용하면 된다.
		- 점을 정규화하고 homography를 씌운 다음에 다시 원래 scale로 원복하는 거임.
- 여기도 RANSAC을 적용할 수가 있다.
	- 4개 점을 가지고 hypothesis를 정하고 진행하면 된다.
	- $Hx-x'$를 outlier를 정하는 식이라 계산하면 된다.
- 이미지 파노라마를 만들 수도 있다.
	- 사진 여러 장을 찍을 때, 3d object 들이 충분히 멀어야 한다. 그래야 각 이미지들 간의 depth discontinuity가 적어져, 무시할 수 있는 수준이 되기 때문이다. 
### 4-3. Epipolar Geometry
- 용어 설명
	- baseline: 두 카메라 중심($C_L, C_R$)을 잇는 직선
	- epipole: baseline이 이미지 평면과 만나는 점.
		- 상대방 카메라의 중심이 내 이미지 평면에 투영된 점.
	- epipolar plane: 공간상의 한 점($P$)과 두 카메라 중심($C_L, C_R$)이 이루는 삼각형 평면.
	- epipolar line: epipolar plane과 이미지 평면이 만나서 생기는 직선.
- 목적: 왼쪽 plane의 점 $x$가 오른쪽에는 어디에 있는지 찾기.
- 이전에는 3D plane을 가정했지만, 여기서는 그런걸 가정하지 않는다.
- 1st 관찰자가 물체를 관찰하는 걸 2nd 관찰자인 우리가 관찰할 수 있다.
- 1st view 위치와 epipole의 위치를 안다고 해도 원본을 복원할 수는 없지만, 정보가 epipolar line 어딘가에 위치함은 알 수 있기에, 연산량이 획기적으로 줄어든다.
	- 원래는 그냥 $NxN$회 연산해서 가장 비슷한 특징점을 찾아야했음.
- matrix $H$ 1개로는 두 image plane 사이의 점 매칭이 불가능하다.
	- depth ambiguity 때문임.
	- 우리가 3D 위치를 가정하지 않았기에 발생함
- epipole은 이미지 밖에 있을 수도 있다.
#### 4-3-1. Two-View Relationship in Epipolar Geometry
- 두 카메라가 calibrate되어 있다고 하자.
	- 이 말은 우리가 카메라의 모든 intrinsic 파라미터를 알고 있고, 이미지 coords.가 sensor coords.에 맞춰져 있는 상황이다.
- essential matrix $E$: 카메라의 내부 파라미터를 알고 있는 상황에서, 두 카메라 사이의 기하학적 관계(회전과 평행이동)만을 나타내는 행렬.
	- $E$는 언제나 존재하고, 항상 찾을 수 있다.
- 공간 상에 주어진 점 $P$에 대해, 1st view plane에 projection된 점 $x$가 있다고 하자. $x$에 $E$ 곱하면 2nd view의 epipolar line $l$에 대응된다.
- 우리가 알고 싶은 건 $P$를 2nd view plane에 projection한 점 $x'$이다.
	- $x'$이 위에서 찾은 line 위에 있다는 걸 이용해, $(x')^T Ex = x'^T l= 0$임을 알 수 있다.
- epipolar plane에서 우리가 찾고 싶은 건 1st, 2nd camera의 optical center에서의 서로의 상대 위치이다.
	- 우리는 두 카메라의 geometric relation을 이미 알고 있기에, 두 optical center의 거리 벡터($t$)와 방향($R$)을 알 수 있다.
	- 이를 이용하면, $RX=X'-t$임을 구할 수 있다.
		- 계산 도중의 $[t]_{\times}(X'-t)$가 $X'$과 수직이기 때문에 inner product는 0
#### 4-3-2. Estimating the Fundamental Matrix
- fundamental matrix $F$: 카메라 정보를 하나도 모르는 상태에서, 순수하게 픽셀 좌표계 사이의 관계를 나타내는 행렬.
	- rank 2 행렬이기에, 처음에는 full rank로 추론하고, 거기서 rank 2 행렬을 유도한다.
	- $(x')^\intercal Fx=0 \Longleftrightarrow xx'f_{11} + \cdots + f_33 = af = 0$
	- 이걸 least square problem으로 풀면 된다.
- 8-point algorithm: numerical stability를 위해, image coords.를 조정해서 중심을 원점으로 만든다. 그 후, RANSAC을 8개 점을 기준으로 수행한다.
