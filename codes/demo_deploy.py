from option import args
import model
import utils
import data.common as common

import torch
import numpy as np
import os
import glob
import cv2

device = torch.device('cpu' if args.cpu else 'cuda')

def deploy(args, sr_model):

    img_ext = '.tif'
    img_lists = glob.glob(os.path.join(args.dir_data, '*'+img_ext))

    if len(img_lists) == 0:
        print("Error: there are no images in given folder!")

    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)

    with torch.no_grad():
        for i in range(len(img_lists)):
            print("[%d/%d] %s" % (i+1, len(img_lists), img_lists[i]))
            # cls_labels = utils.make_labels(args, [os.path.split(img_lists[i])[-1]])
            lr_np = cv2.imread(img_lists[i], cv2.IMREAD_COLOR)
            lr_np = cv2.cvtColor(lr_np, cv2.COLOR_BGR2RGB)  # 简单地读图 (128,128,3)

            if args.cubic_input:
                lr_np = cv2.resize(lr_np, (lr_np.shape[0] * args.scale[0], lr_np.shape[1] * args.scale[0]),
                                interpolation=cv2.INTER_CUBIC)

            lr = common.np2Tensor([lr_np], args.rgb_range)[0].unsqueeze(0) # [1,3,128,128]

            if args.test_block:
                print("================")
                # test block-by-block

                b, c, h, w = lr.shape
                factor = args.scale[0]
                tp = args.patch_size  # [2,128]
                if not args.cubic_input:
                    ip = tp // factor  # [2, 64]
                else:
                    ip = tp

                assert h >= ip and w >= ip, 'LR input must be larger than the training inputs'
                if not args.cubic_input:
                    sr = torch.zeros((b, c, h * factor, w * factor)) 
                else:
                    sr = torch.zeros((b, c, h, w))

                for iy in range(0, h, ip):

                    if iy + ip > h:
                        iy = h - ip
                    ty = factor * iy

                    for ix in range(0, w, ip):

                        if ix + ip > w:
                            ix = w - ip
                        tx = factor * ix

                        # forward-pass
                        lr_p = lr[:, :, iy:iy + ip, ix:ix + ip]
                        lr_p = lr_p.to(device)
                        sr_p = sr_model(lr_p)
                        sr[:, :, ty:ty + tp, tx:tx + tp] = sr_p

            else:

                lr = lr.to(device)
                sr = sr_model(lr)

            sr_np = np.array(sr.cpu().detach())
            final_sr = sr_np[0, :].transpose([1, 2, 0])

            if args.rgb_range == 1:
                final_sr = np.clip(final_sr * 255, 0, args.rgb_range * 255)
            else:
                final_sr = np.clip(final_sr, 0, args.rgb_range)

            final_sr = final_sr.round().astype(np.uint8)
            final_sr = cv2.cvtColor(final_sr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.dir_out, os.path.split(img_lists[i])[-1]), final_sr)


if __name__ == '__main__':
    # args parameter setting
    # UCMerced data
    # args.resume=0
    # args.scale=[4]
    # args.test_block=True
    # args.patch_size=256
    # args.model='HAUNET'
    # print(args)
    # args.pre_train = '../experiment/HAUNET_UCX4/model/model_best.pt'
    # args.dir_data = 'G:\dataset/UCMerced-dataset/test\LR_x4'
    # args.dir_out = '/home/wjq/wjqHD/RSISR/model-zoo/HAUNet_RSISR/experiment/HAUNETx2_UCMerced/results'
    print(args)
    checkpoint = utils.checkpoint(args)
    sr_model = model.Model(args, checkpoint)
    sr_model.eval()

    deploy(args, sr_model)