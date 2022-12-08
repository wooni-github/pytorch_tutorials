# 파이토치 초보자를 위한 repo입니다.

최종적으로 CNN을 이용한 잘 알려진 네트워크들을 구현하는 것을 목표로 하고 있습니다!

ex) Classification, Detection, Segmentation, Regression, GAN, ...


## 기본적인 개발 환경 구성 (RTX3090 : CUDA, CuDNN, Windows10에서 작업)

22.12.08 현재, 파이토치 설치만으로 예제 코드 동작 가능합니다.

```
git clone https://github.com/wooni-github/pytorch_tutorials
cd <다운 위치>
conda env create --file environment.yaml
```

혹은 (torch만 각 환경에 맞게 설정)

```
conda create -n pytorch_tutorials python=3.8
conda activate pytorch_tutorials
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

## [1. 기본적인 텐서 연산](https://github.com/wooni-github/pytorch_tutorials/blob/main/1.Tensors/1.Tensors.md)

기본적인 텐서 생성, 변환, 연산(곱, 행렬곱, 합), 접근에 관한 예제입니다.


