# -*- coding:utf-8 -*- 
class DefaultConfigs(object):
    data_root = '/media/z840/HDD_1/LINUX/GradCAM_callback/Classification-GradCAM-callback/data/NWPU-RESISC45' # 数据集的根目录
    model = 'ResNet18'  # ResNet18, ResNet34, ResNet50, ResNet101, ResNet152 使用的模型
    freeze = False      # 是否冻结卷基层

    seed = 1000         # 固定随机种子
    num_workers = 12    # DataLoader 中的多线程数量
    num_classes = 45    # 分类类别数
    num_epochs = 100
    batch_size = 64
    lr = 0.01           # 初始lr
    width = 256         # 输入图像的宽
    height = 256        # 输入图像的高
    iter_smooth = 50   # 打印&记录log的频率

    resume = True      # 
    checkpoint = 'ResNet18.pth' # 训练完成的模型名

config = DefaultConfigs()
