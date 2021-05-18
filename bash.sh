 CUDA_VISIBLE_DEVICES=1 python train.py --dataset pain -b 1 --prj b2a --direction a_b
 python test.py --dataset pain --nepochs 0 601 20 --prj legacyb16cubic --direction a_b

 python train.py --dataset dess -b 4 --prj a2b --direction a2b --input_nc 1
 python test.py --dataset dess --cuda --nepochs 20 --prj a2b --direction a2b

