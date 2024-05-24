# 复现实验
2023-10-23更新整理！！！

## 原文指标
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx2|HAUNet|34.46|0.9333|0.6437|0.0488|论文|
|UCx3|HAUNet|30.34|0.8476|0.4236|0.0779|论文|
|UCx4|HAUNet|28.06|0.7726|0.2932|0.0997|论文|

## 137复现结果（无随机种子，batch_size=4）
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx2|HAUNet|34.64|0.9349|0.6503|0.0478|137-HAUNETx2_UCMerced|
|UCx3|HAUNet|30.30|0.8487|0.4257|0.0779|137-HAUNETx3_UCMerced|
|UCx4|HAUNet|28.01|0.771|0.2900|0.0999|137-HAUNETx4_UCMerced|

## auto复现结果（无随机种子，batch_size=8）
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx2|HAUNet|34.49|0.9335|0.6452|0.0485|auto-HAUNETx2_UCMerced|
|UCx3|HAUNet|30.20|0.8469|0.4192|0.0788|auto-HAUNETx3_UCMerced|
|UCx4|HAUNet|28.00|0.770|0.2890|0.10045|auto-HAUNETx4_UCMerced|

## auto复现结果（无随机种子，batch_size=4）
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx3|HAUNet|30.36|0.8485|0.4275|0.0777|auto-HAUNETx3_UCMerced_b4|

## auto复现结果（随机种子=1，batch_size=8，lr=0.0008）
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx2|HAUNet|34.56|0.9340|0.6465|0.0483|auto-HAUNETx2_UCMerced_1|
|UCx3|HAUNet|30.25|0.8471|0.4221|0.0786|auto-HAUNETx3_UCMerced_1|
|UCx4|HAUNet|27.97|0.7708|0.2883|0.1007|auto-HAUNETx4_UCMerced_1|

## auto复现结果（随机种子=1，batch_size=8，lr=0.0011）\
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx2|HAUNet|34.503664|0.933529|0.645193|0.048577|auto-HAUNETx2_UCMerced_s1_b8|
|UCx3|HAUNet|30.372846|0.848352|0.426527|0.077667|auto-HAUNETx3_UCMerced_s1_b8|
|UCx4|HAUNet|28.033755|0.772385|0.292654|0.100051|auto-HAUNETx4_UCMerced_s1_b8|

### lr=0.0016 4倍超分 HAUNETx4_UCMerced_s1_b8_lr16
Average: PSNR: 27.910537 dB, SSIM: 0.768060, SAM: 0.101240, QI: 0.990916, SCC: 0.283339

> *按照各种复现实验的结果来看，在接下来的实验中，设置随机种子=1，batch_size=8，lr=0.0011进行实验。*

# 实验
> 接下来的所有实验不同放大倍数保存在相对应的文件夹里面,并且所有的基础实验都在x4倍超分下进行。
## haunet_wo_cim
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx4|HAUNet|27.994505|0.771606|0.290887|0.100234|x4/HAUNET_WO_CIM_UCMerced|

> 去掉CIM模块，直接使用跳跃连接相连

## haunet_cim_v1
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx3|HAUNet|30.242153|0.845377|0.419540|0.078712|auto-x3/HAUNET_CIM_V1_UCMerced|

## haunet_cim_v2
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx4|HAUNet|27.922912 |0.768586|0.285181|0.101022|x4/HAUNET_CIM_V2_UCMerced|

> 将CIM模块替换为MMFU模块

## haunet_cim_v3
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx4|HAUNet|27.596769 |0.756981|0.257580|0.104743|x4/HAUNET_CIM_V3_UCMerced|


## haunet_cim_v4
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx2|HAUNet|34.596980 |0.934544|0.648512|0.047955|x2/HAUNET_CIM_V4_UCMerced|
|UCx3|HAUNet|30.335350 |0.846036|0.424803|0.078084|x3/HAUNET_CIM_V4_UCMerced|
|UCx4|HAUNet|28.065712 |0.772413|0.291110|0.099449|x4/HAUNET_CIM_V4_UCMerced|


## haunet_v1
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx4 |HAUNet|27.744365|0.761731|0.272299|0.103118|x4/HAUNET_V1_UCMerced|


## haunet_v2
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx4 |HAUNet|28.042765|0.772279|0.292276|0.099795|x4/HAUNET_v2_UCMerced|
|UCx3 |HAUNet|30.333430|0.846653|0.426038|0.078025|x3/HAUNET_v2_UCMerced|
|UCx2 |HAUNet|34.483910|0.933892|0.645958|0.048602|x3/HAUNET_v2_UCMerced|

## haunet_v3
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx4 |HAUNet|27.951831|0.769129|0.285571|0.100846|x4/HAUNET_v3_UCMerced|

### lr=15
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx4 |HAUNet|27.622927|0.757825|0.260272|0.104356|x4/HAUNET_v3_lr15_UCMerced|


## haunet_v4
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx4 |HAUNet|27.768876|0.762650|0.272783|0.102916|x4/HAUNET_v3_UCMerced|


## haunet_v5
> haunet_v5因为设计出现问题，暂且搁置

# Train
```bash  
# x4
python demo_train.py --model=HAUNET --dataset=UCMerced --scale=4 --patch_size=192 --ext=img --save=HAUNETx4_UCMerced 
# x3
python demo_train.py --model=HAUNET --dataset=UCMerced --scale=3 --patch_size=144 --ext=img --save=HAUNETx3_UCMerced
# x2
python demo_train.py --model=HAUNET --dataset=UCMerced --scale=2 --patch_size=96 --ext=img --save=HAUNETx2_UCMerced
```
输入LR的大小被裁剪为:48*48，同时有一个数据预处理（包括随机水平、垂直翻转、随机旋转90°，以及添加噪声）。

# Test
```bash
# debug模式
python demo_deploy.py --scale=2 --model=HAUNET --patch_size=128 --test_block --pre_train=/home/wjq/wjqHD/RSISR/model-zoo/HAUNet_RSISR/experiment/HAUNETx2_UCMerced/model/model_best.pt --dir_data=/home/wjq/wjqHD/RSISR/datasets/HAUNet/UCMerced-dataset/test/LR_x2 --dir_out=/home/wjq/wjqHD/RSISR/HAUNet-wjq/experiment/HAUNETx2_UCMerced_debug/results
# x2
python demo_deploy.py --scale=2 --model=HAUNET_WJQ --patch_size=128 --test_block --pre_train=/root/autodl-tmp/experiment/HAUNETWJQx2_UCMerced/model/model_best.pt --dir_data=/root/autodl-tmp/datasets/HAUNet/UCMerced-dataset/test/LR_x2 --dir_out=/root/autodl-tmp/experiment/HAUNETWJQx2_UCMerced/results
# x3
python demo_deploy.py --scale=3 --model=HAUNET --patch_size=192 --test_block --pre_train=/root/autodl-tmp/experiment/HAUNETWJQx2_UCMerced/model/model_best.pt --dir_data=/root/autodl-tmp/datasets/HAUNet/UCMerced-dataset/test/LR_x3 --dir_out=/root/autodl-tmp/experiment/HAUNETWJQx2_UCMerced/results
# x4
python demo_deploy.py --scale=4 --model=HAUNET_WJQ --patch_size=256 --test_block --pre_train=/root/autodl-tmp/experiment/HAUNETWJQx2_UCMerced/model/model_best.pt --dir_data=/root/autodl-tmp/datasets/HAUNet/UCMerced-dataset/test/LR_x4 --dir_out=/root/autodl-tmp/experiment/HAUNETWJQx2_UCMerced/results
```
以`64x64`为block进行测试。2倍时`pathch_size=128`，3倍时`patch_size=192`，4倍时`patch_size=256`。

# 评估指标
```bash
cd metric_scripts 
python calculate_metric.py
```

# 实验结论
1. 无论是使用插值，还是硬train一发，对结果影响不大。
2. 对通道注意力使用残差连接的影响？
3. 将unsample换为转置卷积的影响
4. 在卷积后面添加激活函数的影响
