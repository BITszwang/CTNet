import torch
from torchsummaryX import summary



# import model.ctnet as ctnet
# upscale = 2
# model = ctnet.CTNET(in_nc=3, out_nc=3, nf=64, unf=24, nb=16, scale=upscale)

# import model.hsenet_arch as hsenet
# upscale = 4
# model = hsenet.HSENET(n_feats=64,scale=upscale,n_basic_modules=10,rgb_range=1,n_colors=3)


# import model.dcm_arch as dcm
# upscale = 2
# model = dcm.DIM(scale=upscale,n_colors=3)
#
# if upscale==2:
# # input LR x2, HR size is 256 256
#     summary(model, torch.zeros((1, 3, 128, 128)))
# elif upscale==3:
#     summary(model, torch.zeros((1, 3, 85, 85)))
# else:
#     summary(model, torch.zeros((1, 3, 64, 64)))



# import model.lgcnet_arch as lgcnet
# model = lgcnet.LGCNET(n_colors=3)
# summary(model, torch.zeros((1, 3, 256, 256)))


# import model.srcnn_arch as srcnn
# model = srcnn.SRCNN(n_colors=3)
# summary(model, torch.zeros((1, 3, 256, 256)))


import model.fsrcnn_arch as fsrcnn

upscale =4
model = fsrcnn.FSRCNN(upscale_factor=upscale)

if upscale==2:
# input LR x2, HR size is 256 256
    summary(model, torch.zeros((1, 3, 128, 128)))
elif upscale==3:
    summary(model, torch.zeros((1, 3, 85, 85)))
else:
    summary(model, torch.zeros((1, 3, 64, 64)))


