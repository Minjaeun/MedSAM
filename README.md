# Introduction 

# MedSAM
MedSAM은 SAM(meta, face book회사에서 나온 segment anything 이라는 모델)을 의료 영역에 적용하는 사례를 보여준다. 

## SAM 구조 

    1. image encoder(transformer-based) ; extract image features - output token이 있는데, 이건 기존 ViT 모델에서 쓰던 cls token이랑 유사한 trainable 토큰
        - image encoder안의 vision transformer(1024x1024, high resolution image process 가능)는 masked auto-encoder modeling으로 pretrained 됨. 
        obtained image embedding ; 16x downscaled(64x64)

    2. prompt encoders ; to incorporate user interactions
        - 사용자 입력에 따라 맞춰서 대응할 수 있도록 하기 위함이다. 
        - sam은 네가지 다른 prompt 종류 지운 ; points, boxes, texts, masks
        * sparse prompt ; points, bounding box, text
        * dense promt ; mask


    3. mask decoder ; to generate segmentation results, confidence scores - image embedding, prompt embedding, output token 입력을 기반으로 함.
    특징 ) light weight design. 
    - 두개의 transformer layers는 아래 두개의 head 포함. 
        - dynamic mask prediction head ; output - 세개의 4x downscsaled masks ( 각 각 whole object, part, and subpart of the object에 대응함. )
        - intersection-over-Union(IoU) score regression head




## 위의 모델(sam)을 의료 이미지에 적용했을 때

    a. segmen-anything mode(mask mode) - 두개의 제한점
    - segmentaion results do not have semantic lables
    - clinicians mainly focus on meaningful ROLs in clinial scenarios. ( liver, kidneys, spleen, and lesinos 등. )
    ;즉, useless region 분류를 할 확률이 높다. 

    b. bbox mode - 제대로 잘 검출함
    ; 명확하게 ROI를 확인하고 의미있는 segmentation 결과를 얻을 수 있음. 
    ; radiology에서 흔하게 사용되는 annotation 방법은 longest diameter를 라벨링함.  
    * RECIST (Response Evaluation Criteria in Solid Tumors) annotation - 암 환자의 종양 반응을 평가하기 위해 사용되는 표준화된 척도


    c. point mode - foreground point와 여러번의 background points들을 지정해야 원하는 부위를 segmentation했다. 
    ; 여러번의 predition-correction iteration 이 필요하고 애매하다. 

    따라서 실제 medical image segmentation task에서는 bbox 모두가 실용적이라고 판단. 


## MedSAM

    SAM의 모델 구조( image encoder, prompt encoder, mask decoder) 중 image encoder 부분은 ViT에 기반하는, SAM 모델에서 가장 computational cost가 많이 드는 부분.해당 부분 frozen 상태로 유지. 
    prompt encoder의 pre-trained bounding box encoder부분은 bounding box의 positional information을 충분히 포함하므로 이 부분도 frozen 상태로 유지. 
    mask decoder 부분만 fine-tuning 수행

    - 각 prompt 마다 매번 image embedding을 수행하는 것은 비효율적이므로 trainging image에 대한 embedding은 pre-compute. 
    ; training efficiency  향상. 
    - 기존 mask decoder는 세개의 영역에 대한 mask를 생성했지만 bouinding box prompt가 명확하게 대상을 명시해주므로 하나의 mask만 생성하면 된다. 


    data curation and preprocessing 
        - 33개의 다양한 segmentation task (21개 3D image, 9개 2D image)를 가지는 대규모 데이터셋에 대하여 중점을 두었다.
        
        intensity value 조정
            1. 
            CT image - intensity value [-500, 100] 범위로 clip (대부분의 조직을 포함하는 범위. )
            나머지 이미지 - intensity value를 [0.95th , 99.5th percentiles]
            2. intensity value normalize to the range [0, 255]
            3, resize the images to a uniform size of 256x256x3 

        - randomly split each dataset into 80 and 20 for training and testing
        - segmentation 영역이 100pixel 이하인 것 제외. 
        - SAM은 2D segmentation 모델로 고안되었으므로 3D 이미지들을 2D로 slicing

    training protocol
        - 위에서 전처리한 이미지들을 image encoder에 입력( 256x256x3 ); output image size (3x1024x1024)
        - 학습용 bounding box ; GT mask에서 생성, 랜덤한게 perturbation of 0-20 pixels. - 사용자가 입력하는 bounding box에 유연성을 갖기 위해. 
        - loss function ; Dice loss + Cross-entropy loss (unweighted sum) ; 다양한 segmentation task에서 robust하다고 증명됨. 
        - optimizer ; Adam optimizer, initial learning rate of 1e-5

    evalutaion results
        region overlap ration and boundary consensus evalutaion method
        - DSC (Dice similarity coefficient)
        - NSD ( Normalized Surface Distance, tolerance 1mm)

        SAM과의 성능 비교
        - 3D images
            평균 22.5% DSC 향상, 39.1% NSD 향상
        - 2D images
            평균 17.6% DSC 향상, 18.9% NSD 향상

        SAM은 큰 장기에 대한 성능은 좋았으나 병변 segmentation task에는 좋은 성능을 보이지 못함. 
        - SAM은 3D 작업에 낮은 NSD. 
        - SAM은 2D에서는 비슷한 DSC, NSD는 더 높음. 
        - 주로 경계선 검출 성능이 떨어진다. 
        - lesion 검출 능력이 매우 떨어진다. 
        - SAM, MedSAM은 DSC와 NSD 둘 다 더 우수하다. 



# 환경 설정 (Installation, run)

1. 가상환경 생성
'conda create -n medsam python=3.10 -y'

2. 가상환경 실행
'conda activate medsam'

3. pytorch 2.0 설치

4. git clone 수행

5. install dependency
'cd Med_SAM'
'pip install -e .'




# custom dataset Fine-tuning 

1. check point download.
    SAM check point는 work_dir/SAM directory 에 저장해서 사용하세요. 
    'wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth' 수행시 다운이 가능합니다. 
    기존 SAM에서 pre-train 된 것을 가져온 것입니다. 

2. get medical dataset
    data directory에 원하는 의료 이미지를 저장합니다. 
    fine-tuning demo를 위한 데이터셋은 https://drive.google.com/file/d/18GhVEODbTi17jSeBXdeLQ7vHPdtlTYXK/view?usp=share_link에 있습니다. unzip해서 data 디렉토리 내부에 넣어 사용하세요. 

    이 데이터셋에는 50개의 복부 CT 스캔이 포함되어 있으며, 각 스캔에는 13개의 장기에 대한 주석 마스크가 있습니다. 장기 레이블의 이름은 MICCAI FLARE2022에서 사용 가능합니다. 이 튜토리얼에서는 SAM(Spatial Attention Module)을 담담 세포(담낭) 분할을 위해 파인튜닝할 것입니다.

    만일 fine-tuning을 원하는 데이터셋이 있다면 data directory에 저장하세요. 


3. pre-processing 수행
    'python pre_grey_rgb2D.py'

    만일 fine-tuning을 원하는 2D데이터셋이 있다면 아래와 같이 수행하세요.

    ```bash
    python pre_grey_rbg2D.py -i path_to_image_folder -gt path_to_gt_folder -o path_to_output
    ```

4. model training tutorial 
    finetune_and_inference_tutorial_2D_dataset.ipynb 셀을 실행해서 2D medical dataset tutorial 을 수행하세요. 



# pseudo code
```bash
image encoder
    1. image를 convolutaion2d 연산을 통해 patch Embedding 한다. 
    convolution2d(input_channels = 3, embedding dimension = 768, kernel size = [16, 16], stride = [16, 16], padding = [0, 0])
    2. absolute positional embedding 을 pretrain image size로 초기화한다.
    3. 설정한 깊이 (12)만큼 widnow attention 과 residual propagation을 수행하는 transformer 블럭을 생성한다. 
    4. image encoder의 neck으로 convolution2d, layernormalization 2d, convolutional2d, layernormalization2d를 순차적으로 쌓아 구성한다. 
prompt encoder
    5. convolution2d layer, layer  normalization , activation function layer, convolution2d layer, layer  normalization , activation function layer, convolution2d layer 를 순차적으로 구성한다. 
mask decoder
    6. prompt encoder의 결과와 image encoder의 결과를 사용해 output을 도출한다. 
```

# AI특론 기말과제


## 전처리
    pre_grey_rgb2D.py. 이 때 데이터 위치는 -i annotation으로 지정. 
    다운로드 받은 이미지 파일을 8 :2 로 나눈 뒤 npz 파일로 변경해주는 전처리 pre_grey_rgb2D.py 수행. 
    현재 이미지는 medsam에서 제공하는 데모용 ct라서 기본적으로 -i 지정 안해줘도 default로 들어가서 전처리가 수행됨. 

## fine tuning
    os.makedirs()수행 - 4번째 셀
    'data/demo2d_vit_b' directory가 생성됨. 
    기존의 jpg 파일을 npz 파일로 변환해서 위의 directory에 저장. 
    이 안에 npz 파일들 저장되어있음. 

## inference
### 기존
    test directory 에 있는 실제 gt label의 segmentation 영역에서 bbox를
    추출해 이에 대해 약간의 random한 변형을 주어 잘 segmentation 하는지 수행. 

### 제안 부분
    해당 모델에서 와 다르게 실제 ct영역에서 segmentation auto-labelling으로 사용하려 한다면, CT는 computational tomography, 즉 연속성이 있기 때문에 가까운 거리의 CT 이미지에서 bbox가 어느정도 공유된다고 판단. 
    이로 인해 같은 사람에 한하여 그 전 이미지의 segmentation 결과를 다음 이미지의 bbox로 입력하게 되면 실제 라벨링을 수행할 때 여러장의 segmentation을 위해 bbox를 구해내는 것 보다 한장의 bbox를 통해 전체 ct에서 원하는 결과를 얻을 수 있을 것이라 생각. 
    따라서 현재 test dataset들에 한하여 가장 첫 이미지가 target의 대표 이미지라고 가정하고 gt segmentation을 그대로 가져와서 사용하지 않고 추론한 결과를 바탕으로 다른 이미지를 추론할 수 있도록 한다. 

### 장점
    기존에는 실제 사용상황이라고 한다면 전체 CT 이미지에서 원하는 결과를 위해 여러장의 이미지에 bbox를 일일히 제공해야했지만 현재 제시하는 것은 그 노동력을 줄일 수 있다. 

## Original sam inference  vs medsam inference
    기존에 pretrained된 sam 모델을 가지고 추론한 것과 medical image 에 대하여 fine-tuning한 medsam모델의 추론 결과를 비교한다. 
    이때 위에서 제시한 방법이 gt를 사용해 bbox를 추론한 결과와 유사할수록 실제 효용성이 있을 것으로 예상된다. 

## result 
    위의 제안과 순서를 바탕으로 finetune_and_inference_CT_dataset_SAI에 그 과정을 구현해놓았다. 
    구현 결과, 아래와 같은 결과가 나왔다. 
    
| result | SAM | MedSAM |
|----------|----------|----------|
| 제시방법   | 0.2399| 0.6608|
| gt 사용 | 0.6499 | 0.8921 |

    SAM은 당연하게 fine-tuning된 medsam보다 그 성능이 낮으며, 작은 이미지에 약하다. 반면 MedSAM은 그보다 좋은 성능을 보이는 것을 알 수 있다. 또한 MedSAM의 경우 제시 방법과 gt 사용으로 생성한 bbox의 성능 차이가 0.23정도로, 이는 random하게 생성하는 bbox의 parameter와 더욱 많은 데이터로 학습시 성능이 향상될 것으로 보인다. 
