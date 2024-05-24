### News

**2023-09-14** 为服务器运行程序创建分支autodl

### 复现指标
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx2|HAUNet|34.46|0.9333|0.6437|0.0488|论文|
|UCx2|HAUNet|34.64|0.9349|0.6503|0.0478|137-HAUNETx2_UCMerced|
|UCx2|HAUNet|34.49|0.9335|0.6452|0.0485|auto-HAUNETx2_UCMerced|
|UCx3|HAUNet|30.34|0.8476|0.4236|0.0779|论文|
|UCx3|HAUNet|30.30|0.8487|0.4257|0.0779|137-HAUNETx2_UCMerced|
|UCx4|HAUNet|28.06|0.7726|0.2932|0.0997|论文|
|UCx4|HAUNet|28.01|0.771|0.2900|0.0999|137-HAUNETx2_UCMerced|

和原论文中的指标进行对比，整体来说复现差距很小，复现成功。其中x2超分比原文高0.18。

#### 随机种子42
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx4|HAUNet|27.75|0.7633|0.2717|0.1028|auto-HAUNETx4_UCMerced_42_patch|
|UCx4|HAUNet|27.86|0.7669|0.2809|0.1017|auto-HAUNETx4_UCMerced_42_whole|

### 随机种子1
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx2|HAUNet|34.56|0.9340|0.6465|0.0483|auto-HAUNETx2_UCMerced_1|
|UCx4|HAUNet|27.97|0.7708|0.2883|0.1007|auto-HAUNETx4_UCMerced_1|

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

### 测试一(haunet_wjq.py)
去掉双三次上采样操作
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx2|HAUNet|34.46|0.9333|0.6437|0.0488|论文|
|*UCx2*|HAUNet_wjq|34.50|0.9335|0.6442|0.0485|auto-HAUNETWJQx2_UCMerced|

### 测试二(haunet_no_cim.py)
使用1x1卷积替换掉原论文中的cim模块
|scale|model|PSNR|SSIM|SCC|SAM|location|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|UCx4|HAUNet|28.06|0.7726|0.2932|0.0997|论文|
|UCx4|HAUNet_NO_CIM|27.97|0.7707|0.2883|0.1007|auto-HAUNET_NO_CIMx4_UCMerced|

### Train
```bash
# x4
python demo_train.py --model=HAUNET --dataset=UCMerced --scale=4 --patch_size=192 --ext=img --save=HAUNETx4_UCMerced 
# x3
python demo_train.py --model=HAUNET --dataset=UCMerced --scale=3 --patch_size=144 --ext=img --save=HAUNETx3_UCMerced
# x2
python demo_train.py --model=HAUNET --dataset=UCMerced --scale=2 --patch_size=96 --ext=img --save=HAUNETx2_UCMerced
```
输入LR的大小被裁剪为:48*48，同时有一个数据预处理（包括随机水平、垂直翻转、随机旋转90°，以及添加噪声）。


### Test
```bash
python demo_deploy.py --scale=2 --model=HAUNET --patch_size=128 --test_block --pre_train=/home/wjq/wjqHD/RSISR/model-zoo/HAUNet_RSISR/experiment/HAUNETx2_UCMerced/model/model_best.pt --dir_data=/home/wjq/wjqHD/RSISR/datasets/HAUNet/UCMerced-dataset/test/LR_x2 --dir_out=/home/wjq/wjqHD/RSISR/HAUNet-wjq/experiment/HAUNETx2_UCMerced_debug/results

# auto x2
python demo_deploy.py --scale=2 --model=HAUNET_WJQ --patch_size=128 --test_block --pre_train=/root/autodl-tmp/experiment/HAUNETWJQx2_UCMerced/model/model_best.pt --dir_data=/root/autodl-tmp/datasets/HAUNet/UCMerced-dataset/test/LR_x2 --dir_out=/root/autodl-tmp/experiment/HAUNETWJQx2_UCMerced/results
```
以64x64为block进行测试。2倍时`pathch_size=128`，3倍时`patch_size=192`，4倍时`patch_size=256`。

### 评估指标
```bash
cd metric_scripts 
python calculate_metric.py
```
