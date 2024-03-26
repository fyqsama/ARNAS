# Towards Accurate and Robust Architectures via Neural Architecture Search

Official code of this paper. In the version submitted to CVPR, the naming of the cells is different from that in the paper, but the naming does not affect the running process of the algorithm. In this version, we change the naming to be consistent.

## Requirements
```
Python == 3.7, PyTorch == 1.2.0, torchvision == 0.4.0, cleverhans == 4.0.0
```

## Datasets
Dataset will be downloaded automatically when running the code.

## Architecture search (using small proxy models)
To carry out architecture search using 2nd-order approximation, run
```
cd ARNAS_search/cnn && python train_search.py --unrolled     
```
Please note that we only implement the 2nd-order approximation version

## Training (using full-sized models)

To train the searched architecture in CIFAR-10, run
```
cd ARNAS_train_eval/advrush && python adv_train.py --batch_size 32 --epochs 200 --adv_loss pgd --arch ARNAS
```
If you want to train the architecture searched by yourself, please modify the `--arch` flag, and specify the architecture in `genotypes.py`.

To train the searched architecture in Tiny-ImageNet, run
```
cd train_tiny_imagenet && python adv_train_tinyimagenet.py
```

## Evaluation

To evaluate the trained architecture under white-box attacks, run
```
cd ARNAS_train_eval/eval && python pgd_attack.py
```
We also provide the pretrained model `ARNAS.pth.tar` in File `/ARNAS_train_eval/eval/EXP`, which can reproduce the experimental results of this paper.

Please specify the attack settings in `pgd_attack.py` directly, including:

'--random': True for PGD and False for FGSM  
'--epsilon': Total pertubation scale, 8 / 255 in this paper.  
'--step-size': step-size,  8 / 255 for FGSM, and 2 / 255 for PGD  
'--num-steps': 1 for FGSM, 20 for PGD20, and 100 for PGD100  

To evaluate the trained architecture under AutoAttack, run
```
cd test_AutoAttack && python pgd_attack.py
```