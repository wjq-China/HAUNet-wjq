from option import args
import torch
import numpy as np
import random
import data
import model
import utils
import loss
import trainer
import os 
import datetime
import traceback
from torch.utils.tensorboard import SummaryWriter


def seed_torch(seed=1):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	# torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_torch(args.seed)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    checkpoint = utils.checkpoint(args)
    # writer = SummaryWriter("/home/wjq/wjqHD/RSISR/HAUNet-wjq/experiment/x4/"+ args.save + "/runs")
    writer = SummaryWriter("/root/autodl-tmp/experiment/x4/"+ args.save + "/runs")
    if checkpoint.ok:
        dataloaders = data.create_dataloaders(args)   # dataloaders为一个dict
        sr_model = model.Model(args, checkpoint)
        total = sum([param.nelement() for param in sr_model.parameters()])
        print("Number of parameters: %.4fM" % (total / 1e6))
        sr_loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = trainer.Trainer(args, dataloaders, sr_model, sr_loss, checkpoint)
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        t.ckp.write_log("start_time:" + now + '\n')
        

        while not t.terminate():
            try:
                t.train()
                t.test()
                writer.add_scalar("L1 loss",t.loss.log[-1].numpy(), t.scheduler.last_epoch)
                writer.add_scalar("psnr",t.ckp.log[-1].numpy(), t.scheduler.last_epoch)
            except:
                 traceback.print_exc(file=open("/root/autodl-tmp/experiment/x4/"+ args.save + "/error.log",'a'))
                 break

    end = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    t.ckp.write_log("end_time:" + end + '\n')
    checkpoint.done()
