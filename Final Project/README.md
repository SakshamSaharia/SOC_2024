# DRRN

This project is implementation of the research paper : ["Image Super-Resolution via Deep Recursive Residual Network"](http://cvlab.cse.msu.edu/project-super-resolution.html) as part of the Seasons Of Code 2024 Project on Image Super Resolution.
NOTE: It is suggested that code should only be run on GPU


## Prepare

The images for creating a dataset used for training (**291-image**) or evaluation (**Set5**) can be downloaded from the paper author's [implementation](https://github.com/tyshiwo/DRRN_CVPR17/tree/master/data).

You can also use pre-created dataset files with same settings as the paper.

| Dataset | Scale | Type | Link |
|---------|-------|------|------|
| 291-image | 2, 3, 4 | Train | [Download](https://www.dropbox.com/s/w67yqju1suxejxn/291-image_x234.h5?dl=0) |
| Set5 | 2 | Eval | [Download](https://www.dropbox.com/s/b4a48onyqedx8dz/Set5_x2.h5?dl=0) |
| Set5 | 3 | Eval | [Download](https://www.dropbox.com/s/if01dprb3tzc8jr/Set5_x3.h5?dl=0) |
| Set5 | 4 | Eval | [Download](https://www.dropbox.com/s/cdoxdgz99imy9ik/Set5_x4.h5?dl=0) |

### Generate training dataset

```bash
python generate_trainset.py --images-dir "address_of_directory/Train_291" \
                            --output-path "address_of_directory/Train_291_x234.h5" \
                            --patch-size 31 \
                            --stride 21
```

### Generate test dataset

```bash
python generate_testset.py --images-dir "address_of_directory/Set5" \
                           --output-path "address_of_directory/Set5_x2.h5" \
                           --scale 2
```

## Train

Model weights will be stored in the `--outputs-dir` after every epoch.

```bash
python train.py --train-file "address_of_directory/Train_291_x234.h5" \
                --outputs-dir "address_of_directory/DRRN_B1U9" \
                --B 1 \
                --U 9 \
                --num-features 128 \
                --lr 0.1 \
                --clip-grad 0.01 \
                --batch-size 128 \
                --num-epochs 2 \
                --num-workers 8 \
                --seed 123
```

You can also evaluate using `--eval-file`, `--eval-scale` options during training after every epoch. In addition, the best weights file will be stored in the `--outputs-dir` as a `best.pth`.

```bash
python train.py --train-file "address_of_directory/Train_291_x234.h5" \
                --outputs-dir "address_of_directory/DRRN_B1U9" \
                --eval-file "address_of_directory/Set5_x2.h5" \
                --eval-scale 2 \
                --B 1 \
                --U 9 \
                --num-features 128 \
                --lr 0.1 \
                --clip-grad 0.01 \
                --batch-size 128 \
                --num-epochs 2 \
                --num-workers 8 \
                --seed 123
```

## Evaluate

The pre-trained weights can be downloaded from the following links.

| Model | Link |
|-------|------|
| DRRN_B1U9 | [Download](https://www.dropbox.com/s/1ozete9panliycb/drrn_x234.pth?dl=0) |

```bash
python eval.py --weights-file "address_of_directory/DRRN_B1U9/x234/best.pth" \
               --eval-file "address_of_directory/Set5_x2.h5" \
               --eval-scale 2 \
               --B 1 \
               --U 9 \
               --num-features 128               
```

## Results

\

### Performance comparision on the Set5
Due to gpu and time limitations I ran the code only for 2 epochs , 1 recursive block and only 9 residual units per residual block.
These hyper parameters can be increased to attain much better performance leading closer to the PSNR values obtained in the research paper 
| Eval. Mat | Scale | DRRN_B1U9 (Paper) | DRRN_B1U9 (Mine) |
|-----------|-------|-------|-----------------|
| PSNR | 2 | 37.66 | **36.96** |



Epoch-wise PSNR in my code :

|-----------|-------|-------|-----------------|
| Eval. Mat | Scale | Epoch Number |
| PSNR | 2 | 0| **36.89** |
| PSNR | 2 | 1| **36.96** |


## References

1. [https://github.com/tyshiwo/DRRN_CVPR17](https://github.com/tyshiwo/DRRN_CVPR17)
