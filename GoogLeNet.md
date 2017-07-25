
Convolutaion Neural Networks
=====================================
> 2017.07.25.  
> CNN 정리노트 4

---

## GoogLeNet

GoogLeNet에서는 망의 깊이 및 넓이가 모두 커지고, 중간에 분기되는 부분도 있고, “인셉션”이라는 생소한 모듈이 등장한다. 하지만, 이 모든 것들이 구글의 연구팀들이 최초로 개발한 것은 아니며, 이들 역시 타 연구자들의 연구와 자신들의 연구를 융합하여 발전시킨 결과이다.  
구글은 자신들의 구조를 설계함에 있어 크게 2개의 논문을 참조하고 있으며, 그 중 인셉션 모듈 및 전체 구조에 관련된 부분은 싱가포르 국립 대학의 “Min Lin”이 2013년에 발표한 “Network In Network” 구조를 더욱 발전 시킨 것이다.


## NIN(Network In Network) 구조와 설계 철학  

NIN은 말 그대로 네트워크 속의 네트워크를 뜻한다.

NIN 설계자는 CNN의 convolutional layer가 local receptive field에서 feature를 추출해내는 능력은 우수하지만,  filter의 특징이 linear하기 때문에 non-linear한 성질을 갖는 feature를 추출하기엔 어려움이 있으므로, 이 부분을 극복하기 위해 feature-map의 개수를 늘려야 하는 문제에 주목했다. 필터의 개수를 늘리게 되면 연산량이 늘어나는 문제가 있다.

그래서 NIN 설계자는 local receptive field 안에서 좀 더 feature를 잘 추출해낼 수 있는 방법을 연구하였으며, 이를 위해 micro neural network를 설계하였다. 이들은 convolution을 수행하기 위한 filter 대신에 MLP(Multi-Layer Perceptron)를 사용하여 feature를 추출하도록 하였으며, 그 구조는 아래 그림과 같다.

![](./images/NIN.png)

CNN은 filter의 커널을 입력 영상의 전체 영역으로 stride 간격만큼씩 옮겨가면서 연산을 수행한다. 반면에 NIN에서는 convolution 커널 대신에 MLP를 사용하며, 전체 영역을 sweeping 하면서 옮겨가는 방식은 CNN과 유사하다.

MLP를 사용했을 때의 장점은 convolution kernel 보다는 non-linear 한 성질을 잘 활용할 수 있기 때문에 feature를 추출할 수 있는 능력이 우수하다는 점이다. 또한 1x1 convolution을 사용하여 feature-map을 줄일 수 있도록 하였으며, 이 기술은 GoogLeNet의 인셉션에 그대로 적용이 된다.


NIN 이름이 붙은 이유는 망을 깊게 만들기 위해, mlpconv layer를 여러 개를 쌓아 사용을 하기 때문이며, 그래서 네트워크 안에 네트워크가 있다는 개념이 만들어지게 되었다. GoogLeNet에서도 인셉션 모듈을 총 9개를 사용하기 때문에 개념적으로는 NIN과 맥이 닿아 있다고 볼 수 있다.


![](./images/mlpconv.png)


NIN 구조가 기존 CNN과 또 다른 점은 CNN의 최종단에서 흔히 보이는  fully-connected neural network이 없다는 점이다. (위 그림 참조)  
Fully-connected NN 대신에 최종단에 “Global average pooling”을 사용하였다. 이는 앞에서 효과적으로 feature-vector를 추출하였기 때문에, 이렇게 추출된 vector 들에 대한 pooling 만으로도 충분하다고 주장을 하고 있다. Average pooling 만으로 classifier 역할을 할 수 있기 때문에 overfitting의 문제를 회피할 수 있고, 연산량이 대폭 줄어드는 이점도 얻을 수 있다.

## 1x1 Convolution이란?

Convolution은 local receptive field의 개념을 적용하기 때문에 7x7, 5x5, 3x3과 같이 주변 픽셀들의 정보를 같이 활용을 한다. 그런데 괴상한(?) 이름의 1x1 convolution 이라는 개념이 나온다.   
1x1 convolution을 하는 결정적인 이유는 차원을 줄이는 것이다. GoogLeNet 소개 논문에 나오는 것처럼, Hebbian principle(Neurons that fire together, wire together)에 의해 차원의 줄일 수 있다.   
1x1 convolution을 수행하면, 여러 개의 feature-map으로부터 비슷한 성질을 갖는 것들을 묶어낼 수 있고, 결과적으로 feature-map의 숫자를 줄일 수 있으며, feature-map의 숫자가 줄어들게 되면 연산량을 줄일 수 있게 된다. 또한 연산량이 줄어들게 되면, 망이 더 깊어질 수 있는 여지가 생기게 된다.

![](./images/cccp.png)

위 그림에서 “C2 > C3”의 관계가 만들어지면, 차원을 줄이는 것과 같은 효과를 얻을 수 있기 때문에, GoogLeNet을 포함한 최신 CNN 구조에서는 1x1 convolution을 많이 사용한다.

1x1 convolution은 처음에는 개념적으로 쉽게 와닿지 않는다. 논문이나 설명 글을 참고할 때 1x1 convolution을 1-layer fully-connected neural network이라고도 하는데, 그 이유는 1x1 convolution이 fully-connected와 동일한 방식이기 때문이다.  
만약에 입력 feature-map c2의 갯수가 4이고, 출력 feature-map c3의 갯수가 2인 경우를 가정해보면, 1x1 convolution은 아래 그림과 같이 표현할 수 있다.

![](./images/1x1_conv.png)


결과적으로 보면 4개의 feature-map으로부터 입력을 받아, 학습을 통해 얻어진 learned parameter를 통해 4개의 feature-map이 2개의 feature-map으로 결정이 된다. 즉, 차원이 줄어들게 되며, 이를 통해 연산량을 절감하게 된다.   
또한, neuron에는 활성함수로 RELU를 사용하게 되면, 추가로 non-linearity를 얻을 수 있는 이점도 있다.

## 구글의 인셉션(Inception)

원래 1x1 convolution, 3x3 및 5x5 convolution, 3x3 max pooling을 나란히 놓는 구조를 고안하였다. 다양한 scale의 feature를 추출하기에 적합한 구조가 된다.  
하지만 곧 문제에 부딪치게 된다. 3x3과 5x5 convolution은 연산량의 관점에서 보면, expensive unit(값 비싼 대가를 치러야 하는 unit)이 된다.  
망의 깊이가 깊지 않을 때는 큰 문제가 아니나, 망의 깊이와 넓이가 깊어지는 GoogLeNet 구조에서는 치명적인 결과가 될 수도 있다.

![](./images/inception.png)

그래서 아래의 그림과 같이 3x3 convolution과 5x5 convolution의 앞에, 1x1 convolution을 cascade 구조로 두고, 1x1 convolution을 통해 feature-map의 개수(차원)를 줄이게 되면, feature 추출을 위한 여러 scale을 확보하면서도, 연산량의 균형을 맞출 수 있게 된다.  
GoogLeNet의 22 layer까지 깊어질 수 있는 것도 따지고 보면, 1x1 convolution을 통해 연산량을 조절할 수 있었기 때문에, 이렇게 깊은 수준까지 망을 구성할 수 있게 된 것이다.  
3x3 max pooling에 대해서도 1x1 convolution을 둔다.

![](./images/inception_reduction.png)

NIN에서는 MLP를 이용하여 non-linear feature를 얻어내는 점에 주목을 했지만, MLP는 결국 fully-connected neural network의 형태이고, 구조적인 관점에서도 그리 익숙하지 못하다.  
반면에 구글은 기존의 CNN 구조에서 크게 벗어나지 않으면서도 효과적으로 feature를 추출할 수 있게 되었다. 요약하면, 인셉션 모듈을 통해 GoogLeNet 은 AlexNet에 비해 망의 깊이는 훨씬 깊은데 free parameter의 수는 1/12 수준이고 전체 연산량도 AlexNet에 비해 적다는 것을 알 수가 있다.  
참고로 GoogLeNet에는 총 9개의 인셉션 모듈이 적용되어 있다.

## GoogleLeNet의 핵심 철학 및 구조

GoogLeNet 의 핵심 설계 철학은 주어진 하드웨어 자원을 최대한 효율적으로 이용하면서도 학습 능력은 극대화 할 수 있도록 깊고 넓은 망을 갖는 구조를 설계하는 것이다.
인셉션 모듈에 있는 다양한 크기의 convolution kernel(그림에서 파란색 부분)을  통해 다양한 scale의 feature를 효과적으로 추출하는 것이 가능해졌다. 또한 인셉션 모듈 내부의 여러 곳에서 사용되는 (위 그림의 노란색 부분) 1x1 convolution  layer를 통해, 연산량을 크게 경감시킬 수 있게 되어, 결과적으로 망의 넓이와  깊이를 증가시킬 수 있는 기반이 마련 되었다. 이 인셉션 모듈을 통해 NIN (Network-in-Network) 구조를 갖는 deep CNN 구현이 가능하게 되었다.

GoogLeNet에는 아래 그림과 같이, 총 9개의 인셉션 모듈이 적용이 되었다. 그림에서 빨간색 동그라미가 인셉션 모듈에 해당이 된다. 망이 워낙 방대하여 작은 그림으로 표현하기에 아쉬움이 있다.

![](./images/googlenet.png)

위 그림에서 
* 파란색 유닛은 convolutional layer
* 빨간색은 max-pooling 유닛
* 노란색 유닛은 Softmax layer
* 녹색은 기타 function  

인셉션 모듈을 나타내는 동그라미 위에 있는 숫자는 각 단계에서 얻어지는 feature-map의 수를 나타낸다.

![](./images/googlenet_cost.png)

GoogLeNet의 각 layer의 구조는 위 표와 같다. 위 표는 얼핏 보기에는 복잡해 보이지만, 조금만 살펴보면 그 내용을 정확하게 이해할 수 있다.



l  Patch size/stride: 커널의 크기와 stride 간격을 말한다. 최초의 convolution에 있는 7x7/2의 의미는 receptive field의 크기가 7x7인 filter를 2픽셀 간격으로 적용한다는 뜻이다.

l  Output size: 얻어지는 feature-map의 크기 및 개수를 나타낸다. 112x112x64의 의미는 224x224 크기의 이미지에 2픽셀 간격으로 7x7 filter를 적용하여 총 64개의 feature-map이 얻어졌다는 뜻이다.

l  Depth: 연속적인 convolution layer의 개수를 의미한다. 첫번째 convolution layer는 depth가 1이고, 두번째와 인셉션이 적용되어 있는 부분은 모두 2로 되어 있는 이유는 2개의 convolution을 연속적으로 적용하기 때문이다.

l  #1x1: 1x1 convolution을 의미하며, 그 행에 있는 숫자는 1x1 convolution을 수행한 뒤 얻어지는 feature-map의 개수를 말한다. 첫번째 인셉션 3(a)의 #1x1 위치에 있는 숫자가 64인데 이것은 이전 layer의 192개 feature-map을 입력으로 받아 64개의 feature-map이 얻어졌다는 뜻이다. 즉, 192차원이 64차원으로 줄어들게 된다.

l  #3x3 reduce: 이것은 3x3 convolution 앞쪽에 있는 1x1 convolution 을 의미하여 마찬가지로 인셉션 3(a)의 수를 보면 96이 있는데, 이것은 3x3 convolution을 수행하기 전에 192차원을 96차원으로 줄인 것을 의미한다.

l  #3x3: 1x1 convolution에 의해 차원이 줄어든 feature map에 3x3 convolution을 적용한다. 인셉션 3(a)의 숫자 128은 최종적으로 1x1 convolution과 3x3 convolution을 연속으로 적용하여 128개의 feature-map이 얻어졌다는 뜻이다.

l  #5x5 reduce: 해석 방법은 “#3x3 reduce”와 동일하다.

l  #5x5: 해석 방법은 “#3x3”과 동일하다. #5x5는 좀 더 넓은 영역에 걸쳐 있는 feature를 추출하기 위한 용도로 인셉션 모듈에 적용이 되었다.

l  Pool/proj: 이 부분은 max-pooling과 max-pooling 뒤에 오는 1x1 convolution을 적용한 것을 의미한다. 인셉션 3(a) 열의 숫자 32 는 max-pooling과 1x1 convolution을 거쳐 총 32개의 feature-map이 얻어졌다는 뜻이다.

l  Params: 해당 layer에 있는 free parameter의 개수를 나타내며, 입출력 feature-map의 數에 비례한다. 인셉션 3(a) 열에 있는 숫자 159K는 총 256개의 feature-map을 만들기 위해 159K의 free-parameter가 적용되었다는 뜻이다.

l  Ops: 연산의 수를 나타낸다. 연산의 수는 feature-map의 수와 입출력 feature-map의 크기에 비례한다. 인셉션 3(a)의 단계에서는 총 128M의 연산을 수행한다.


위 설명에 따라 표에 있는 각각의 숫자들의 의미를 해석해 보면, GoogLeNet의 구조를 좀 더 친숙하게 이해할 수 있다.