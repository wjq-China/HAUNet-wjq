# 复现实验
2024-06-12 创建dev分枝

## 原文指标
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx2|HAUNet|34.46|0.9333|0.6437|0.0488|论文|
|UCx3|HAUNet|30.34|0.8476|0.4236|0.0779|论文|
|UCx4|HAUNet|28.06|0.7726|0.2932|0.0997|论文|

## 复现指标
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx2|HAUNet|34.600366|0.934406|0.648207|0.048025|HAUNet_UCMerced| 
|UCx3|HAUNet|30.365873|0.848231|0.426694|0.077634|HAUNet_UCMerced_0621| 
|UCx4|HAUNet|27.982857|0.770506|0.288954|0.100645|HAUNet_UCMerced| 
|UCx4|HAUNet|27.973446| 0.769437|0.285880|0.100629|HAUNet_UCMerced_0622_l16| 
|UCx4|HAUNet|28.001716| 0.770383|0.288050|0.100411|HAUNet_UCMerced_0623_b8| 

> 按照lr=0.0011, batchsize=8 进行训练

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
python demo_deploy.py --scale=2 --model=HAUNET --patch_size=128 --test_block --pre_train=/root/autodl-tmp/experiment/x2/HAUNET_UCMerced/model/model_best.pt --dir_data=/root/autodl-tmp/datasets/UCMerced-dataset/test/LR_x2 --dir_out=/root/autodl-tmp/experiment/x2/HAUNET_UCMerced/results
# x2
python demo_deploy.py --scale=2 --model=HAUNET_WJQ --patch_size=128 --test_block --pre_train=/root/autodl-tmp/experiment/HAUNETWJQx2_UCMerced/model/model_best.pt --dir_data=/root/autodl-tmp/datasets/HAUNet/UCMerced-dataset/test/LR_x2 --dir_out=/root/autodl-tmp/experiment/HAUNETWJQx2_UCMerced/results
# x3
python demo_deploy.py --scale=3 --model=HAUNET --patch_size=192 --test_block --pre_train=/root/autodl-tmp/experiment/x3/HAUNET_UCMerced/model/model_best.pt --dir_data=/root/autodl-tmp/datasets/UCMerced-dataset/test/LR_x3 --dir_out=/root/autodl-tmp/experiment/x3/HAUNET_UCMerced/results
# x4
python demo_deploy.py --scale=4 --model=HAUNET --patch_size=256 --test_block --pre_train=/root/autodl-tmp/experiment/x4/HAUNET_UCMerced_0623_b8/model/model_best.pt --dir_data=/root/autodl-tmp/datasets/UCMerced-dataset/test/LR_x4 --dir_out=/root/autodl-tmp/experiment/x4/HAUNET_UCMerced_0623_b8/results
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
