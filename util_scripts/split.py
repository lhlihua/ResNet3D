import cv2
import os

# 视频数据集存储位置
video_path = 'E:\\train_data\\UCF101\\UCF-101\\'
# 生成的图像数据集存储位置
save_path = 'E:\\train_data\\jpgs\\'
# 如果文件路径不存在则创建路径
if not os.path.exists(save_path):
    os.mkdir(save_path)
# 获取动作列表
action_list = os.listdir(video_path)
# 遍历所有动作
for action in action_list:
    if action.startswith(".")==False:
        if not os.path.exists(save_path+action):
            os.mkdir(save_path+action)
        video_list = os.listdir(video_path+action)
        #遍历所有视频
        for video in video_list:
            prefix = video.split('.')[0]
            if not os.path.exists(os.path.join(save_path, action, prefix)):
                os.mkdir(os.path.join(save_path, action, prefix))
            save_name = os.path.join(save_path, action, prefix) + '/'
            video_name = video_path+action+'/'+video
            # 读取视频文件
            # cap为视频的帧
            cap = cv2.VideoCapture(video_name)
            # fps为帧率
            fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_count = 0
            for i in range(fps):
                ret, frame = cap.read()
                if ret:
                    # 将帧画面写入图片文件中
                    cv2.imwrite(save_name+str(10000+fps_count)+'.jpg',frame)
                    fps_count += 1

