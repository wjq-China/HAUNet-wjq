"""
双分支: SAB+CA(通道注意力)
flops: 4.7028G, params: 2.3283M
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def make_model(args, parent=False):
    return HAUNet(up_scale=args.scale[0], width=96, enc_blk_nums=[5,5],dec_blk_nums=[5,5],middle_blk_num=10)

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
    

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)



class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
class Reconstruct(nn.Module):
    def __init__(self, scale_factor):
        super(Reconstruct, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        if self.scale_factor!=1:
            x = nn.Upsample(scale_factor=self.scale_factor)(x)
        return x
   
class CA(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(c, c // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(c // reduction, c, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):  # (32,64,32,32)
        
        y = self.avg_pool(x)  # (32,64,1,1)
        y = self.conv_du(y)  # (32,64,1,1)
        return x * y  # (32,64,32,32)   
    
class MMFU(nn.Module):
    def __init__(self, c):
        super().__init__()
        # 1x1卷积
        # print("c is: ",c)
        self.conv = nn.Conv2d(3*c, c, kernel_size=1)
        # 通道注意力
        self.ca = CA(c)
        pass
    
    def forward(self, x):
        y = self.conv(x)
        y = self.ca(y)
        return y

class lateral_nafblock_wjq(nn.Module): 
    def __init__(self, c):
        super().__init__()

        self.MMFU_0 = MMFU(c)
        self.MMFU_1 = MMFU(c)
        self.MMFU_2 = MMFU(c)

    def forward(self, encs):
        enc0, enc1, enc2 = encs[0], encs[1], encs[2]
        
        outs = []

        # 对encoder0进行处理
        """
        首先将enc1和enc2上采样,然后三个尺度进行cat,
        再1x1卷积降维,最后使用一个通道注意力
        """
        enc1_0 = nn.Upsample(scale_factor=2)(enc1)
        enc2_0 = nn.Upsample(scale_factor=4)(enc2)
        y0 = torch.cat([enc0, enc1_0, enc2_0], dim=1)
        out0 = self.MMFU_0(y0)
        out0 = out0 + enc0
        outs.append(out0)

        # 对encoder1进行处理
        """
        首先将enc0下采样,enc2上采样
        """
        enc0_1 = nn.Upsample(scale_factor=0.5)(enc0)
        enc2_1 = nn.Upsample(scale_factor=2)(enc2)
        y1 = torch.cat([enc0_1, enc1, enc2_1], dim=1)
        out1 = self.MMFU_1(y1)
        out1 = out1 + enc1
        outs.append(out1)

        # 对encoder2 进行处理
        """
        首先将enc0, enc1进行下采样
        """
        enc0_2 = nn.Upsample(scale_factor=0.25)(enc0)
        enc1_2 = nn.Upsample(scale_factor=0.5)(enc1)
        y2 = torch.cat([enc0_2, enc1_2, enc2], dim=1)
        out2 = self.MMFU_2(y2)
        out2 = out2 + enc2
        outs.append(out2)

        return outs

class CAB(nn.Module):
    def __init__(self, c, compress_ratio=3, reduction=16):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            # nn.Conv2d(c, c//compress_ratio, 3, 1, 1),
            # nn.GELU(),
            # nn.Conv2d(c, c, 3, 1, 1),
            CA(c)
        )
    def forward(self, x):
        return self.cab(x)


class S_CEMBlock_wjq(nn.Module):
    def __init__(self, c, DW_Expand=2, num_heads=3, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.num_heads = num_heads

        self.norm1 = LayerNorm2d(c)  # 第一个layernorm

        # SAB
        self.qkv = nn.Conv2d(c, c*3, kernel_size=1) # 1x1卷积升
        self.qkv_dwconv = nn.Conv2d(c*3, c*3, kernel_size=3, stride=1, padding=1, groups=c*3)
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)
        # CAB
        self.cab = CAB(c)

        # SAB和CAB的两个系数
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate >0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate >0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta2 = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
       
        self.norm2 = LayerNorm2d(c)  # 第二个LayerNorm
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sg = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel//2, c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        pass

    def forward(self, inp):
        x = inp
        x = self.norm1(x)

        # 通道注意力
        outc = self.cab(x)
        
        b, c, h, w = x.shape
        
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)  # 通道维数一分为三
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 通道注意力
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        qs = q.clone().permute(0, 1, 3, 2)  # 空间注意力
        ks = k.clone().permute(0, 1, 3, 2)
        vs = v.clone().permute(0, 1, 3, 2)


        qs = torch.nn.functional.normalize(qs, dim=-1)
        ks = torch.nn.functional.normalize(ks, dim=-1)
        attns = (qs @ ks.transpose(-2, -1)) * self.temperature2
        attns=self.relu(attns)
        attns = self.softmax(attns)
        outs = (attns @ vs)
        outs = outs.permute(0, 1, 3, 2)  # 空间注意力的输出
        outs = rearrange(outs, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # outc = rearrange(outc, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        xc = self.dropout1(outc)
        xs = self.dropout2(outs)

        y = inp + xc * self.beta + xs * self.beta2 

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
        pass

class HAUNet(nn.Module):

    def __init__(self, up_scale=4, img_channel=3, width=180, middle_blk_num=10, enc_blk_nums=[5,5], dec_blk_nums=[5,5], heads = [1,2,4],):
        """_summary_

        Args:
            up_scale (int, optional): 放大倍数. Defaults to 4.
            img_channel (int, optional): 输入通道数. Defaults to 3.
            width (int, optional): _description_. encoder和decoder的通道个数 to 180. 实际传入的是96
            middle_blk_num (int, optional): _description_. Defaults to 10.
            enc_blk_nums (list, optional): 单个encoder里blocks的个数为5. Defaults to [5,5].
            dec_blk_nums (list, optional): 单个decoder里blocks的个数为5. Defaults to [5,5].
            heads (list, optional): _description_. Defaults to [1,2,4].
        """
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()


        chan = width  # width传入为96
        ii=0
        for numii in range(len(enc_blk_nums)):
            num = enc_blk_nums[numii]
            if numii < 1:
               self.encoders.append(
                    nn.Sequential(
                        *[S_CEMBlock_wjq(chan, num_heads=heads[ii]) for _ in range(num)],
                        nn.Conv2d(chan, chan, 3, 1, 1)
                    )
                )
            else:
                self.encoders.append(
                    nn.Sequential(
                        *[S_CEMBlock_wjq(chan, num_heads=heads[ii]) for _ in range(num)],
                        nn.Conv2d(chan, chan, 3, 1, 1)
                    )
                )
            self.downs.append(
                nn.Conv2d(chan, chan, 2, 2)
            )
            ii+=1

        self.lateral_nafblock = lateral_nafblock_wjq(chan)
        self.enc_middle_blks = \
            nn.Sequential(
                *[S_CEMBlock_wjq(chan, num_heads=heads[ii]) for _ in range(middle_blk_num // 2)],
                nn.Conv2d(chan, chan, 3, 1, 1)
            )
        self.dec_middle_blks = \
            nn.Sequential(
                *[S_CEMBlock_wjq(chan, num_heads=heads[ii]) for _ in range(middle_blk_num // 2)],
                #nn.Conv2d(chan, chan, 3, 1, 1)
            )
        ii=0
        for numii in range(len(dec_blk_nums)):
            num = dec_blk_nums[numii]
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose2d(chan, chan, kernel_size=2, stride=2)
                )
            )
            # chan = chan // 2
            if numii < 1:
                self.decoders.append(
                    nn.Sequential(
                        *[S_CEMBlock_wjq(chan, num_heads=heads[1 - ii]) for _ in range(num)],
                        #nn.Conv2d(chan, chan, 3, 1, 1)
                    )
                )
            else:
                self.decoders.append(
                    nn.Sequential(
                        *[S_CEMBlock_wjq(chan, num_heads=heads[1 - ii]) for _ in range(num)],
                        #nn.Conv2d(chan, chan, 3, 1, 1)
                    )
                )
            ii += 1
        self.dec_blk_nums=dec_blk_nums
        self.padder_size = 2 ** len(self.encoders)
        self.up = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale ** 2, kernel_size=3, padding=1, stride=1,
                      groups=1, bias=True),
            nn.PixelShuffle(up_scale)
        )
        self.up_scale = up_scale

    def forward(self, inp):
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x) + x
            encs.append(x)  # Encoder的输出结果存起来
            x = down(x)

        x = self.enc_middle_blks(x) + x    # 第三个encoder
        encs.append(x)  # 三个encoder的输出
        outs = self.lateral_nafblock(encs)  # 中间模块的输出
        x = outs[-1]
        x = self.dec_middle_blks(x) + x
        outs2 = outs[:2]
        for decoder, up, enc_skip in zip(self.decoders, self.ups, outs2[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x) + x
        
        x = self.up(x)
        x = x + inp_hr

        return x

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

if __name__ == '__main__':
    import time
    from thop import profile
    net = HAUNet(up_scale=4, width=96, enc_blk_nums=[5,5],dec_blk_nums=[5,5],middle_blk_num=10).cuda()
    torch.cuda.reset_max_memory_allocated()
    x = torch.rand(1, 3, 64, 64).cuda()
    y = net(x)
    # 获取模型最大内存消耗
    max_memory_reserved = torch.cuda.max_memory_reserved(device='cuda') / (1024 ** 2)

    print(f"模型最大内存消耗: {max_memory_reserved:.2f} MB")
    flops, params = profile(net, (x,))
    print('flops: %.4f G, params: %.4f M' % (flops / 1e9, params / 1000000.0))
    net = net.cuda()
    x = x.cuda()
    timer = Timer()
    timer.tic()
    for i in range(100):
        timer.tic()
        y = net(x)
        timer.toc()
    print('Do once forward need {:.3f}ms '.format(timer.total_time * 1000 / 100.0))