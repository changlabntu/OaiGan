import torch
from .unetclean import UNetClean


class BoneSegModel():
    def __init__(self):
        net = UNetClean(4)
        net.load_state_dict(torch.load('utils/bone_segmentation/clean_femur_tibia_cartilage.pth'))
        self.net = net.cuda()


if __name__ == '__main__':
    from utils.bone_segmentation.bonesegmentation import BoneSegModel
    m = BoneSegModel()
    import torch
    from utils.make_config import load_config
    path = load_config('config.ini', 'Path')

    from dataloader.data import get_test_set

    root_path = path['dataset']  # "dataset/"
    test_set = get_test_set(root_path + 'painpickedgood', 'a_b', mode='test')