 

## Project Structures
    ├── dataset                    
        ├── A
            ├── 001.png
            ├── ...
        ├── B
            ├── 001.png
            ├── ...
    ├── models              
        ├── pix2pix.py
        ├── .....
    ├── train.py
    ├── test.py
    ├── README.md
    ├── .....

### Training
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset pain -b 1 --prj baseline --direction a_b
```
run the dataset under dataset/$$$/
```bash
--dataset $$$
```
name of the project
```bash
--prj
```
translate the images from the folder dataset/DATASET/A to dataset/DATASET/B
```bash
--direction A_B
```
run under Pytorch without lightning
```bash
--legacy
```

### Testing (still working on it)
```bash
 python test.py --dataset pain --nepochs 0 601 20 --prj baseline --direction a_b
```

### Refer to this when add new GANs
https://github.com/nocotan/pytorch-lightning-gans
