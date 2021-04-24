from vit_retri.models import DSHNet
import torch

class Args():
    def __init__(self):
        self.model_type='ViT-B_16'
        self.img_size=448
        self.pretrained_dir='/data/fine-grained/ViT-B_16.npz'
        self.hash_bit = 64

if __name__ == "__main__":
    args = Args()
    model = DSHNet(args)
    ckpt = torch.load('output/dch_64_joint/dch_64_joint_best.ckpt')
    model.load_state_dict(ckpt)
