 CUDA_VISIBLE_DEVICES=1 python train.py --dataset pain -b 1 --prj b2a --direction a_b
 python test.py --dataset pain --nepochs 0 601 20 --prj eff --direction a_b

# cyclegan

 CUDA_VISIBLE_DEVICES=1 python train_cycle.py --dataset pain -b 1 --prj cycle0 --direction aregis1small_b