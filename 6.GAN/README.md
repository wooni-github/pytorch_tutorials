
<br>

# GAN (Generative Adversarial Networks)

딥러닝의 꽃이라고 불리는 `GAN` 입니다. 

최근에는 너무나도 대단한 모델들이 나오고 있지만, 간단한 기초적 예제들만 살펴보겠습니다. 저도 아직은 입문자라, 깊은 범위까지는 모르지만요!

## [GAN : MNIST MLP](https://github.com/wooni-github/pytorch_tutorials/blob/main/6.GAN/6.1.MNIST_MLP_GAN/README.md)

[Training] 예제코드 [pytorch_tutorials/6.GAN/6.1.MNIST_MLP_GAN/MNIST_MLP_GAN_Train.py](https://github.com/wooni-github/pytorch_tutorials/blob/main/6.GAN/6.1.MNIST_MLP_GAN/MNIST_MLP_GAN_Train.py)

[Inference] 예제코드 [pytorch_tutorials/6.GAN/6.1.MNIST_MLP_GAN/MNIST_MLP_GAN_Test.py](https://github.com/wooni-github/pytorch_tutorials/blob/main/6.GAN/6.1.MNIST_MLP_GAN/MNIST_MLP_GAN_Test.py)

우선 앞서 `MNIST Image Classification`에서 `CNN`이 아닌 `MLP`로 분류를 진행했던 것과 비슷한 [[Link]](https://github.com/wooni-github/pytorch_tutorials/blob/main/3.SimpleExamples/3.3.MNIST_MLP/README.md) `MLP`를 이용한 생성 모델 예제입니다.

![MLP_GAN](6.1.MNIST_MLP_GAN/MNIST_MLP_GAN.gif)

<br>

가장 기초적인 생성 모델로, 예제 이미치저럼 학습이 진행됨에 따라 이미지가 사실처럼 보이긴 하지만, 아직 제약이 많습니다.

대표적으로 특정 숫자(범주/클래스)의 이미지만을 생성해 낼 수 없죠.

