import cv2
import torch
import os
import torch.nn.functional as F
from PIL import Image as img

# 读取类别词典
def read_classes(path):
    with open(path, "r+") as f:
        lines = f.readlines()
    classes = {}
    for line in lines:
        c_id = int(line.split()[0])-1
        c_name = line.split()[1]
        classes[c_id] =c_name
    return classes

# video 为 视频文件的地址
def predictAction(model, video, spatial_transform, output_topk):
  print('predict')
  model.eval()
  cap = cv2.VideoCapture(video)
  clip = []
  clips = []
  class_names = read_classes("E:/resnet_data/txt/classInd.txt")
  class_pre_name = ''
  score_pre = 0
  with torch.no_grad():
    while True:
      # try:
        ret, frame = cap.read()
        if ret:
          tmp = img.fromarray(frame,"RGB")
          tmp = spatial_transform(tmp)
          clip.append(tmp)
          # 每2个16帧进行一次预测
          if len(clip) == 16:
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            clips.append(clip)
            if len(clips) >= 2:
              clips = torch.stack(clips)
              outputs = model(clips)
              outputs = F.softmax(outputs, dim=1).cpu()
              sorted_scores, locs = torch.topk(outputs,
                                        k=min(output_topk, len(class_names)))
              for i in range(len(sorted_scores[0])):
                num = locs[0][i].item()
                score = sorted_scores[0][i].item()
                class_pre_name = class_names[num]
                score_pre = score
              # cv2.putText(frame, class_names[num], (20,20+top_margin), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
              #             (0, 0, 255), 1)
              # cv2.putText(frame, "prob: %.4f" % score, (20, 40+top_margin),
              #             cv2.FONT_HERSHEY_SIMPLEX, 0.6,
              #             (0, 0, 255), 1)
            # 播放预测后的视频
              clips=[]
            clip=[]
          cv2.putText(frame, class_pre_name, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                          (0, 0, 255), 1)
          cv2.putText(frame, "prob: %.4f" % score_pre, (20, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                          (0, 0, 255), 1)
          cv2.imshow('result',frame)
          cv2.waitKey(30)
        else:
          break
      # except:
      #   print('error in capping')
  cap.release()
  cv2.destroyAllWindows()
  