#CUDA_VISIBLE_DEVICES=1 python train.py --dataset pain -b 1 --prj bysubjectl_d20_100 --lamb 20 --lamb_b 100 --direction aregis1_b --bysubject --resize 286
#CUDA_VISIBLE_DEVICES=1 python train.py --dataset pain -b 1 --prj bysubjectl_d40_100 --lamb 40 --lamb_b 100 --direction aregis1_b --bysubject --resize 286
#CUDA_VISIBLE_DEVICES=1 python train.py --dataset pain -b 1 --prj bysubjectl_d60_100 --lamb 60 --lamb_b 100 --direction aregis1_b --bysubject --resize 286
#CUDA_VISIBLE_DEVICES=1 python train.py --dataset pain -b 1 --prj bysubjectl_d80_100 --lamb 80 --lamb_b 100 --direction aregis1_b --bysubject --resize 286
#CUDA_VISIBLE_DEVICES=1 python train.py --dataset pain -b 1 --prj bysubjectl_d100_100 --lamb 100 --lamb_b 100 --direction aregis1_b --bysubject --resize 286


CUDA_VISIBLE_DEVICES=0 python test.py --dataset pain --nepochs 0 601 20 --prj bysubjectl_d20_100 --direction a_b --resize 286
CUDA_VISIBLE_DEVICES=0 python test.py --dataset pain --nepochs 0 601 20 --prj bysubjectl_d40_100 --direction a_b --resize 286
CUDA_VISIBLE_DEVICES=0 python test.py --dataset pain --nepochs 0 601 20 --prj bysubjectl_d60_100 --direction a_b --resize 286
CUDA_VISIBLE_DEVICES=0 python test.py --dataset pain --nepochs 0 601 20 --prj bysubjectl_d80_100 --direction a_b --resize 286
CUDA_VISIBLE_DEVICES=0 python test.py --dataset pain --nepochs 0 601 20 --prj bysubjectl_d100_100 --direction a_b --resize 286
CUDA_VISIBLE_DEVICES=0 python test.py --dataset pain --nepochs 0 601 20 --prj bysubjectl_d20_40 --direction a_b --resize 286
CUDA_VISIBLE_DEVICES=0 python test.py --dataset pain --nepochs 0 601 20 --prj bysubjectl_d40_40 --direction a_b --resize 286
CUDA_VISIBLE_DEVICES=0 python test.py --dataset pain --nepochs 0 601 20 --prj bysubjectl_d60_40 --direction a_b --resize 286
CUDA_VISIBLE_DEVICES=0 python test.py --dataset pain --nepochs 0 601 20 --prj bysubjectl_d80_40 --direction a_b --resize 286
CUDA_VISIBLE_DEVICES=0 python test.py --dataset pain --nepochs 0 601 20 --prj bysubjectl_d100_40 --direction a_b --resize 286