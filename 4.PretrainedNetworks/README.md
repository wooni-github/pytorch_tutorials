
<br>

# Pretrained networks

파이토치에서 제공하는 다양한 네트워크의 pretrained weights를 이용한 예제입니다.

## [Image classification : ImageNet dataset Inference](4.1.PretrainedClassification/)

pretrained 네트워크를 이용하여 ImageNet 데이터셋에 대한 이미지 분류를 수행하는 예제입니다.

![pertrained_classification](4.1.PretrainedClassification/pretrained_classification.png)


## [Image segmentation : **F**ully **C**onvolutional **N**etwork (**FCN**) Inference](4.2.PretrainedSegmentation/)

pretrained 네트워크 (`FCN`)를 이용하여 이미지 세그먼테이션을 수행하는 예제입니다.


<br>

**Image inference**

|Input|Segmentation result|Visualize|
|---|---|---| 
|![input_image](4.2.PretrainedSegmentation/test_image.jpg)|![result_image](4.2.PretrainedSegmentation/test_image_result.png)|![overlay_image](4.2.PretrainedSegmentation/test_image_overlay.png)|

<br>

**Video inference**

![Video](4.2.PretrainedSegmentation/seq1_result.gif)

<br>

## [**H**uman **P**ose **E**stimation (**HPE**) : keypoint R-CNN Inference](4.3.PretrainedHPE/)

사람의 관절을 추정하는 pretrained 네트워크 `Keypoint R-CNN` (backbone : `ResNet-50-FPN`) 의 예제입니다.

<br>

**Image inference**

|Input|HPE result|
|---|---|
|![input_image](4.3.PretrainedHPE/test_image.jpg)|![result_image](4.3.PretrainedHPE/test_image_result.png)|

<br>

**Video inference**

![Video](4.3.PretrainedHPE/test_video_result.gif)






