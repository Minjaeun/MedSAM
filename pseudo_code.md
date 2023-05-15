#MedSAM
MedSAM은 SAM(meta, face book회사에서 나온 segment anything 이라는 모델)을 의료 영역에 적용하는 사례를 보여준다. 

##SAM 구조 
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


##위의 모델(sam)을 의료 이미지에 적용했을 때

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


##MedSAM

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



#환경 설정

1. 가상환경 생성
'conda create -n medsam python=3.10 -y'

2. 가상환경 실행
'conda activate medsam'

3. pytorch 2.0 설치

4. git clone 수행

5. install dependency
'cd MedSAM'
'pip install -e .'




#custom dataset Fine-tuning 

1. check point download.
    SAM check point는 work_dir/SAM directory 에 저장되어있습니다. 기존 SAM에서 pre-train 된 것을 가져온 것입니다. 

2. get medical dataset
    data directory에 원하는 의료 이미지를 저장합니다. 
    fine-tuning demo를 위한 데이터셋은 data/FALRE22Train에 저장되어있습니다. 

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