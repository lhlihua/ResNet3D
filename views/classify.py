# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'classify.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(993, 661)
        self.selectVideoButton = QtWidgets.QPushButton(Dialog)
        self.selectVideoButton.setGeometry(QtCore.QRect(810, 10, 111, 51))
        self.selectVideoButton.setObjectName("selectVideoButton")
        self.startButton = QtWidgets.QPushButton(Dialog)
        self.startButton.setGeometry(QtCore.QRect(810, 600, 111, 51))
        self.startButton.setObjectName("startButton")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(760, 100, 191, 31))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.openCameraButton = QtWidgets.QPushButton(Dialog)
        self.openCameraButton.setGeometry(QtCore.QRect(810, 160, 111, 51))
        self.openCameraButton.setObjectName("openCameraButton")
        self.selectModelButton = QtWidgets.QPushButton(Dialog)
        self.selectModelButton.setGeometry(QtCore.QRect(810, 230, 111, 51))
        self.selectModelButton.setObjectName("selectModelButton")
        self.ModelLabel = QtWidgets.QLabel(Dialog)
        self.ModelLabel.setGeometry(QtCore.QRect(760, 310, 191, 31))
        self.ModelLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.ModelLabel.setObjectName("ModelLabel")
        self.selectDictButton = QtWidgets.QPushButton(Dialog)
        self.selectDictButton.setGeometry(QtCore.QRect(810, 390, 111, 51))
        self.selectDictButton.setObjectName("selectDictButton")
        self.dictLabel = QtWidgets.QLabel(Dialog)
        self.dictLabel.setGeometry(QtCore.QRect(770, 470, 191, 31))
        self.dictLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.dictLabel.setObjectName("dictLabel")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 741, 641))
        self.label_2.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.selectVideoButton.setText(_translate("Dialog", "选择视频"))
        self.startButton.setText(_translate("Dialog", "开始识别"))
        self.label.setText(_translate("Dialog", "视频名称"))
        self.openCameraButton.setText(_translate("Dialog", "打开摄像头"))
        self.selectModelButton.setText(_translate("Dialog", "选择模型"))
        self.ModelLabel.setText(_translate("Dialog", "模型名称"))
        self.selectDictButton.setText(_translate("Dialog", "选择辞典"))
        self.dictLabel.setText(_translate("Dialog", "辞典名称"))

