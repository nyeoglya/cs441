- goal: understanding camera models
- Homogeneous coordinates
- Geometric primitives / transforms
- Pinhole camera model and projective geometry
- Camera calibration
### 3-1. Preliminaries
#### 3-1-1. Homogeneous Coordinates
- inhomogeneous coordinates: $x = [x,y]^T \in \mathbb{R}^2$.
- homogeneous coordinates: $\bar{x}=[x,y,1]^T$로 2차원을 표현.
	- $\tilde{x} = [\tilde{x}, \tilde{y}, \tilde{w}]^T = \tilde{w} \cdot [x, y, 1]^T = \tilde{w} \cdot \bar{x} \in \mathbb{P}^2$. 여기서, $\mathbb{P}^2 = \mathbb{R}^3 - [0,0,0]^T$.
	- \[0,0,0\]는 존재하지 않는다.
- ideal points: homogeneous point 중에서 $\tilde{w}=0$인 점. 무한한 위치에 있는 점 나타내며 보통 방향을 표시한다.
- 정리하자면, vector 중에서 scale로만 구분되는 점을 동일하게 취급하는 좌표계이다.
- line equation: $\bar{x}^T \cdot \tilde{l} = ax+by+c = 0$이다. 여기서 $\tilde{l}=[a,b,c]^T$이다.
	- 두 선을 교차하는 점은 $\tilde{x} = \tilde{l}_1 \times \tilde{l}_2$이다. 만약, 두 선이 평행하면, $\tilde{x}$는 ideal point이다.
	- 두 점을 지나는 선은 $\tilde{l} = \tilde{x}_1 \times \tilde{x}_2$이다.
	- $\tilde{l}=[0,0,1]^\intercal$이면, line at infinity이다.
#### 3-1-2. 2D Transformations
- translation: 평행이동.
$$\bar{x}' = \begin{pmatrix}I & t \\ 0^T & 1\end{pmatrix} \bar{x} = \bar{x} + \begin{pmatrix}t_x \\t_y \\ 0\end{pmatrix}$$
- euclidean: 위의 translation에서 회전을 더한 것으로, 위에서 $I$가 회전행렬 $R$로 바뀐다.
- similarity: 위의 euclidean에서 scale을 더한 것으로, 위에서 $R$이 $sR$로 바뀐다. 여기서 $s \in \mathbb{R}$이다.
- affine: $sR$ 부분을 임의의 matrix $A$로 바꾼 것이다. 아직까지는 parallel을 보존한다.
- projective: 전체 matrix를 통째로 $H$로 바꾼 것이다. 이것은 homography라고도 불린다.
	- 값을 normalize해서 자유도 하나를 줄일 수 있어서 freedom은 8이다. 정확히는 (3,3)-element를 1로 만드는 거다.
	- straight -> straight / parallel -> may not parallel
- projective는 행렬 맨 아래줄의 값이 존재하기에, 평행이 보존되지 않는다.
#### 3-1-2. 3D
- point: 기본적으로 2D point와 비슷하다.
- line: 두 점을 이용해 유일하게 표현된다.
	- inhomogeneous coords.에서는 $r=(1-t)p+tq$
	- homogeneous coords.에서는 $r=a\tilde{p}+b\tilde{q}$
- plane: 2차원에서 line이 그랬던 것처럼, homogeneous 좌표 1개로 표현 가능하다.
	- $\bar{x}^T \tilde{m} = ax+by+cz+d=0$
	- normalized plane eqn.: $\tilde{m}=[nx,ny,nz,d]^T=[n,d]^T$ where $||n||=1$
	- $\tilde{m}=[0,0,0,1]^\intercal$이면, plane at infinity이다.
- 3D rotation matrix: used to form a camera model.
#### 3-1-3. 3D to 2D projection
- orthographic projection: 직교 투영. 직육면체 영역을 직사각형 영역으로 만든다.
- perspective projection: 사영 투영. 사각뿔 영역을 직사각형 영역으로 만든다.
### 3-2. Geometric Camera Models
- 카메라는 기본적으로 눈이랑 똑같이 작동하도록 디자인한다.
- image: 이미지는 3차원 대상을 2차원에 projection한 것이다. 이미지는 다음의 2가지 정보를 담는다.
	- photometric (이 수업에서는 다루지 않음)
	- geometric: 위치, 점, 선, 곡선과 같은 기하학적 특징들.
- 카메라 자체의 parameter가 있고, 이러한 파라미터들이 실제 좌표와 카메라 상의 투영 좌표의 관게를 결정한다.
- intrinsic parameters: 카메라의 내부 구조에 의해 결정
	- principal point
	- focal length
	- skew coefficient
- extrinsic parameters: 카메라의 외부 구조(위치 등)에 의해 결정
	- camera position
	- camera orientation
#### 3-2-1. Pinhole Model
- blackbox + tiny pinhole + image plane(상이 그려지는 곳).
- image plane에는 이미지가 뒤집혀서 그려진다.
	- sensor plane: image plane의 물리적 용어. 디지털 카메라에서 이미지 평면이 맺히는 바로 그 위치에 빛을 실제로 감지하는 이미지 센서(CMOS 같은거)가 놓여있는 평면
- focal length: pinhole과 image plane 사이의 거리
- virtual image plane: pinhole로부터 focal length만큼 거리에 있는 상자 밖의 가상의 이미지 plane으로 뒤집히지 않은 이미지가 투영된다.
- pinhole 모델은 다음과 같다.
$$\begin{pmatrix} u\\v\\1\end{pmatrix} \propto K P \begin{pmatrix} R & t\\ 0^T & 1\end{pmatrix}\begin{pmatrix} x\\y\\z\\1\end{pmatrix} = K[R,t] \begin{pmatrix} x\\y\\z\\1\end{pmatrix} $$
	- 3D rotation과 translation은 world coordinate가 camera coordinate(보통 pinhole이 원점)와 다를 수 있으니 그걸 맞춰주기 위함.
	- $P$은 sensor plane으로 projection하는 matrix.
	- $K$는 2차원 변환으로 image plane, sensor plane 연결함.
- pinhole model을 잘 이해하려면 매우 간단한 상황부터 가정하면서, 하나씩 assumption을 지워나가면 된다. 이 과정에서 $P$는 다른 matrix에 흡수되어 사라진다.
- intrinsic assumption
	1. skew가 없다.
		- skew는 이미지 센서의 x, y축이 완전히 직교하지 않아 생긴 왜곡이다. 즉, x에 대해 y가 얼마나 전단 변형이 일어났는지에 대한 값이다.
	2. 센서가 전부 unit aspect ratio를 갖는다.
		- 이미지 센서 픽셀이 정사각형이 아니라 직사각형이면 aspect ratio가 달라진다.
	3. optical center가 원점이다.
- extrinsic assumption
	1. 카메라의 회전, 위치가 원점이다.
### 3-3. Geometric Camera Calibration
- camera calibration: camera model을 추론하는 것
- camera 모델은 결국 $K[R,t]$이다. 전체 행렬이 3x4 matrix이기 때문에, image coords. world coords. 쌍의 적당한 집합을 갖고 있으면 된다.
	- 크기와 구조를 잘 알고 있는 패턴이 담긴 큐브를 준비한다. 큐브의 한쪽 점의 world coords만 알아도 모든 좌표를 다 알아낼 수 있다.
- matrix의 degree of freedom이 11이고, 각 점 좌표가 방정식 2개를 만들기 때문에(결과값이 2차원이라 2개임), 6개 이상의 3D-2D correspondence가 필요.
- 이론적으로 모든 값이 정확하게 맞아떨어져야 하지만, 실제로는 그렇지 않기 때문에, squared error를 최소화하도록 하는 것을 목적으로 한다.
$$\min_M\sum_i (m_{11}x_i+m_{12}y_i+m_{13}z_i+m_{14}-\bar{u}_i (\cdots))^2+(\cdots)^2=\min_m m^T A^T A m$$
- 여기서 $m$은 $M$을 flatten해서 만든 12차원 벡터이고, $A$는 곱해지는 행렬로 실제 데이터 값으로 결정된다.
- 이 최적화 문제는 $A^TA$를 eigen decomposition해서 가장 작은 eigenvalue를 찾으면 된다.
	- rayleigh quotient 문제(least square 문제)임.
#### 3-3-1. Estimation of Intrinsic & Extrinsic Parameters
- 계산된 3x4 matrix $M=K[R,t]$을 이용해 intrinsic, extrinsic 매개변수를 알려면 아래의 과정을 수행해야 한다.
- 우선, $M=[G,h]$라 하자. 그러면 $G=KR$, $h=Kt$이다. 여기서 $GG^T = KR(KR)^T = KK^T$이다. 이제 $GG^T=[g_{ij}]$를 normalize하고 행렬이 대칭이라는 것을 이용하여 양변을 비교하면 intrinsic parameter를 바로 얻을 수 있다.
- 이제, 우리는 G와 K를 알고 있다. 그러면 extrinsic parameter를 $R=K^{-1}G$, $t=K^{-1}h$와 같이 얻을 수 있다.
