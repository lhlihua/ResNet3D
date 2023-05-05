import sys
import classify
import torch
import cv2
import torch.nn.functional as F
from camera import Camera
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PIL import Image as img
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
sys.path.append('D:/3D-ResNets-PyTorch')
from main import get_opt, generate_model, load_pretrained_model, get_predict_spatial_transform
from predict import read_classes

class MainDialog(QDialog):
  def __init__(self, parent=None):
    super(QDialog, self).__init__(parent)
    self.ui = classify.Ui_Dialog()
    self.ui.setupUi(self)
    self.ui.selectVideoButton.clicked.connect(self.loadVideo)
    self.ui.startButton.clicked.connect(self.startRecognize)
    self.ui.selectDictButton.clicked.connect(self.loadDict)
    self.ui.selectModelButton.clicked.connect(self.loadModel)
    self.ui.openCameraButton.clicked.connect(self.operateCamera)
    self.camera_thread = Camera()
    self.camera_thread.set_cam_number(0) # 设置摄像头数目 0表示1个  cap.open(0)  表示打开默认摄像头
    # 连接Camera中的信号与本类中的槽(receive)进行展示
    self.camera_thread.sendFrameAndRet.connect(self.receive)
    self.videoPath = ''
    self.modelPath=''
    self.openCameraFlag = 0   # 摄像头默认为不开启状态
    self.startInfe = False
    self.opt = get_opt()
    self.model = generate_model(self.opt)
    if self.opt.pretrain_path:
      self.model = load_pretrained_model(self.model, self.opt.pretrain_path, self.opt.model,
                                      self.opt.n_finetune_classes)
    self.spatial_transform = get_predict_spatial_transform(self.opt)
    # 初始化中间处理变量
    self.clip = []
    self.clips = []
    self.class_pre_name = ''
    self.score_pre = 0
    self.class_names = ''
  
  # 加载视频路径    
  def loadVideo(self):
    fileName,_ = QFileDialog.getOpenFileName(self, '打开视频文件', directory='E:/ml_project/C3D/UCF-101', filter='Vedio(*.avi)')
    self.ui.label.setText(fileName.split('/')[-1])
    self.videoPath = fileName
    
  # 加载模型路径
  def loadModel(self):
    modelPath, _ = QFileDialog.getOpenFileName(self,'打开模型文件',directory='E:/resnet_data/checkmodel', filter='Model(*.pth)')
    self.modelPath = modelPath
    self.ui.ModelLabel.setText(modelPath.split('/')[-1])
    # 重新选择模型，需要重新加载模型
    self.startInfe = False
    
  # 加载辞典路径
  def loadDict(self):
    dictPath, _ = QFileDialog.getOpenFileName(self,'打开辞典文件',directory='E:/resnet_data/txt', filter='Model(*.txt)')
    self.dictPath = dictPath
    self.ui.dictLabel.setText(dictPath.split('/')[-1])

  # 开始动作识别
  def startRecognize(self):
    if self.videoPath == '' or self.modelPath == '':
      print('video or model can not be null')
      return
    self.startInfe = True
    print('recognizing')
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.model = self.get_resume_model(self.modelPath, self.model)
    self.model.eval()
    self.model = self.model.to('cpu')
    cap = cv2.VideoCapture(self.videoPath)
    self.class_names = read_classes(self.dictPath)
    # 中间处理变量初始化
    self.clip = []
    self.clips = []
    self.class_pre_name = ''
    self.score_pre = 0
    with torch.no_grad():
      while True:
        ret, frame = cap.read()
        if ret:
          self.showInLabel(frame, ret)
          cv2.waitKey(50)
        else: 
          break
  # 展示结果函数
  def showInLabel(self, frame, ret):
    if ret:
      # 如果没有进行预测
      if not self.startInfe:
        height, width = frame.shape[:2]
        pixmap = QImage(frame, width, height, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(pixmap)
        # 适应label与视频的比
        ratio = max(width/self.ui.label_2.width(), height/self.ui.label_2.height())
        pixmap.setDevicePixelRatio(ratio)
        self.ui.label_2.setAlignment(Qt.AlignCenter)
        self.ui.label_2.setPixmap(pixmap)
      else:
        self.clip.append(self.spatial_transform(img.fromarray(frame,"RGB")))
        # 每2个16帧进行一次预测
        if len(self.clip) == 16:
          self.clip = torch.stack(self.clip, 0).permute(1, 0, 2, 3)
          # clip = clip.to(device)
          self.clips.append(self.clip)
          if len(self.clips) >= 2:
            self.clips = torch.stack(self.clips)
            # clips = clips.to(device)
            outputs = self.model(self.clips)
            outputs = F.softmax(outputs, dim=1).cpu()
            sorted_scores, locs = torch.topk(outputs,
                                      k=min(1, len(self.class_names)))
            for i in range(len(sorted_scores[0])):
              num = locs[0][i].item()
              score = sorted_scores[0][i].item()
              self.class_pre_name = self.class_names[num]
              self.score_pre = score
            self.clips = []
          self.clip = []
        cv2.putText(frame, self.class_pre_name, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
        cv2.putText(frame, "prob: %.4f" % self.score_pre, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
        height, width = frame.shape[:2]
        pixmap = QImage(frame, width, height, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(pixmap)
        # 适应label与视频的比
        ratio = max(width/self.ui.label_2.width(), height/self.ui.label_2.height())
        pixmap.setDevicePixelRatio(ratio)
        self.ui.label_2.setAlignment(Qt.AlignCenter)
        self.ui.label_2.setPixmap(pixmap)
        
      
  def get_resume_model(self,resume_path, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model
  
  def operateCamera(self):
    # 如果摄像头是开启的话，则关闭摄像头
    if self.openCameraFlag:
      self.camera_thread.close_camera()
      self.ui.openCameraButton.setText('开启摄像头')
      self.openCameraFlag = False
    # 否则开启摄像头
    else:
      self.camera_thread.open_camera()
      self.ui.openCameraButton.setText('关闭摄像头')
      self.openCameraFlag = True
  
  def receive(self, frame, ret):
    # print('receive')
    self.showInLabel(frame, ret)
    
    
if __name__ == '__main__':
  app = QApplication(sys.argv)
  dlg = MainDialog()
  dlg.show()
  sys.exit(app.exec_())