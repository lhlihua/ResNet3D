from PyQt5.Qt import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QImage
import cv2
import time

class Camera(QObject):
  # sendPicture = pyqtSignal(QImage)
  sendFrameAndRet = pyqtSignal(object,int)
  
  def __init__(self, parent=None):
    super(Camera, self).__init__(parent)
    self.thread = QThread()
    self.moveToThread(self.thread)
    self.timer = QTimer()
    self.init_timer()
    self.camera_num = 0
    
  def init_timer(self):
    self.timer.setInterval(30)
    self.timer.timeout.connect(self.display)
    
  def set_cam_number(self, n):
    self.camera_num = n
    
  def open_camera(self):
    print('open camera')
    self.cap = cv2.VideoCapture()
    self.cap.set(4, 480)
    self.cap.set(3, 640)
    self.cap.open(self.camera_num)
    self.timer.start()
    self.thread.start()
  
  def close_camera(self):
    print('close camera')
    self.cap.release()
    
  def display(self):
    ret, frame = self.cap.read()
    time.sleep(0.05)      # 耗时操作
    # showImage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
    # self.sendPicture.emit(showImage)
    self.sendFrameAndRet.emit(frame,ret)
