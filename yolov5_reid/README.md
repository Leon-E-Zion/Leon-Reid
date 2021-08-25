## Leon  七夕节  累了困了  好不起来了


## 2021/8/25--更新相关操作：
### 1-环境准备：
————将model_final.pth置于目录yolov5_reid/model_yaml下
————将yolov5s.pt置于目录yolov5_reid/weights下
### 2- 其他相关材料
————model_final.pth的下载地址：
————yolov5s.pt的下载地址

### 3-开始预测：
##### 选择目标人物阶段：
——将包含目标任务的视频移动到yolov5_reid目录，命名test.mp4
——在yolov5_reid文件夹中运行 P-Net-Reid.py 即可对摄像头进行相关的检测，记住在该视频中目标人物的序号
——运行yolov5_reid\fast_reid\demo的person_bank文件，输入需要跟踪的目标，程序会自动生成相关的特征提取信息
##### 追踪：
——将需要进行追踪判断的视频移动到yolov5_reid目录，命名test.mp4
——在yolov5_reid文件夹中运行 P-Net-Reid.py 即可对摄像头进行相关的检测，被检测到的目标下方会显示‘xx-figures'等图标，xx为其在选择目标人物阶段提取人物特征时的锁定序号
<——也可向 P-Net-Reid.py 在命令行中填入相关参数   开始预测>

