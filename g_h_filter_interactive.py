import sys
import os
from PyQt5 import QtGui
from PyQt5 import QtCore
import functools
import numpy as np
import random as rd
from numpy.random import randn
import matplotlib
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget,QFrame,QMainWindow,QSizePolicy)
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time
import threading



def setCustomSize(x, width, height):
    sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(x.sizePolicy().hasHeightForWidth())
    x.setSizePolicy(sizePolicy)
    x.setMinimumSize(QtCore.QSize(width, height))
    x.setMaximumSize(QtCore.QSize(width, height))

''''''

class CustomMainWindow(QMainWindow):

    def __init__(self):

        super(CustomMainWindow, self).__init__()

        # Define the geometry of the main window
        self.setGeometry(300, 300, 2400, 1200)
        self.setWindowTitle("g_h_filter window")

        # Create FRAME_A
        self.FRAME_A = QFrame(self)
        self.FRAME_A.setStyleSheet("QWidget { background-color: %s }" % QtGui.QColor(210,210,235,255).name())
        self.LAYOUT_A = QGridLayout()
        self.FRAME_A.setLayout(self.LAYOUT_A)
        self.setCentralWidget(self.FRAME_A)
        self.createFilterGroupBox()
        self.LAYOUT_A.addWidget(self.filterGroupBox, *(0,0))

        # Place the matplotlib figure
        self.myFig = CustomFigCanvas(100,1000)
        self.LAYOUT_A.addWidget(self.myFig, *(0,1))

        
        #establish signals and slots
        self.slider_x.valueChanged.connect(self.myFig.valueChanged_x)
        self.slider_dx.valueChanged.connect(self.myFig.valueChange_dx)
        self.slider_h.valueChanged.connect(self.myFig.valueChange_h)
        self.slider_g.valueChanged.connect(self.myFig.valueChange_g)
        self.show()

    ''''''

    def createFilterGroupBox(self):
            self.filterGroupBox = QGroupBox("h_f_filter")
            self.xLabel = QLabel("x")
            self.dxLabel = QLabel("dx")
            self.hLabel = QLabel("h")
            self.gLabel = QLabel("g")
            self.slider_x = QSlider(Qt.Horizontal, self.filterGroupBox)
            self.slider_x.setRange(0,100)
            self.slider_x.setValue(50)
            self.slider_dx = QSlider(Qt.Horizontal, self.filterGroupBox)
            self.slider_dx.setRange(0,100)
            self.slider_dx.setValue(50)
            self.slider_h = QSlider(Qt.Horizontal, self.filterGroupBox)
            self.slider_h.setRange(0,100)
            self.slider_h.setValue(50)
            self.slider_g = QSlider(Qt.Horizontal, self.filterGroupBox)
            self.slider_g.setRange(0,100)
            self.slider_g.setValue(50)

            self.xValue = QLabel("0")
            self.dxValue = QLabel("0")
            self.hValue = QLabel("0")
            self.gValue = QLabel("0")
            #establish signals and slots
            self.slider_x.valueChanged.connect(self.valueChanged_x)
            self.slider_dx.valueChanged.connect(self.valueChange_dx)
            self.slider_h.valueChanged.connect(self.valueChange_h)
            self.slider_g.valueChanged.connect(self.valueChange_g)

            #Label for understanding the filter
            self.textlabel = QLabel("#Initialization \
                #1. Initialize the state of the filter \
                #2. Initialize our belief in the state \
                #Predict \
                #1. Use system behavior to predict state at the next time step \
                #2. Adjust belief to account for the uncertainty in prediction \
                #Update \
                #1. Get a measurement and associated belief about its accuracy \
                #2. Compute residual between estimated state and measurement \
                #3. New estimate is somewhere on the residual line")
            self.textlabel.setSizePolicy( QSizePolicy.Expanding, QSizePolicy.Preferred )
            self.textlabel.setWordWrap(True)
            layout = QGridLayout()
            layout.addWidget(self.xLabel, 0, 0)
            layout.addWidget(self.dxLabel, 1, 0)
            layout.addWidget(self.hLabel, 2, 0)
            layout.addWidget(self.gLabel, 3, 0)
            layout.addWidget(self.slider_x, 0, 1)
            layout.addWidget(self.slider_dx, 1, 1)
            layout.addWidget(self.slider_h, 2, 1)
            layout.addWidget(self.slider_g, 3, 1)
            layout.addWidget(self.xValue, 0, 2)
            layout.addWidget(self.dxValue, 1, 2)
            layout.addWidget(self.hValue, 2, 2)
            layout.addWidget(self.gValue, 3, 2)

            layout.addWidget(self.textlabel, 4,0,1,3)
            layout.setRowStretch(3, 1)
            self.filterGroupBox.setLayout(layout)
            self.show()

    
    def valueChanged_x(self):
        self.x = self.slider_x.value()*4 - 200
        self.xValue.setText(str(self.x))
        

    def valueChange_dx(self):
        self.dx = self.slider_dx.value() - 50
        self.dxValue.setText(str(self.dx))
        

    def valueChange_h(self):
        self.h = 0.0199 * self.slider_h.value() + 0.01
         
        self.hValue.setText(str("{:.3f}".format(self.h)))
        

    def valueChange_g(self):
        self.g = 0.005 * self.slider_g.value()
        self.gValue.setText(str("{:.3f}".format(self.g)))
    

''' End Class '''




''''''

class CustomFigCanvas(FigureCanvas):

    def __init__(self,xlim,ylim):
        print(matplotlib.__version__)

        # The data
        self.data = []
        self.xlim = xlim
        self.filter_x = 0
        self.filter_dx = 0
        self.filter_h = 0
        self.filter_g = 0

        # The window
        self.fig = Figure(figsize=(15,15), dpi=100)
        self.ax1 = self.fig.add_subplot(111)


        # self.ax1 settings
        self.ax1.set_xlabel('time')
        self.ax1.set_ylabel('raw data')
        self.ax1.set_xlim(0, self.xlim - 1)
        self.ax1.set_ylim(0-ylim, ylim)
        
        #create line1 for filter result
        self.line1 = Line2D([], [], color='blue')
        self.ax1.add_line(self.line1)
       
        #create line2 for measurements
        self.line2 = Line2D([], [], color='yellow')
        self.ax1.add_line(self.line2)

        #generate measured data 
        self.x = np.arange(0, self.xlim, 1)
        self.measuredData = self.gen_data(5,5,self.xlim,50)

        FigureCanvas.__init__(self, self.fig)

    def plot(self):
        self.line2.set_data(self.x,self.measuredData)
        self.line1.set_data(self.x,self.data)
        self.draw()
        
    def valueChanged_x(self,value):
        self.filter_x = value*4 - 200
        print("self.filter_x:",self.filter_x)
        self.reCalc()
        
    def valueChange_dx(self,value):
        self.filter_dx =value - 50
        self.reCalc()

    def valueChange_h(self,value):
        self.filter_h = 0.0199 * value + 0.01
        self.reCalc()

    def valueChange_g(self,value):
        self.filter_g = 0.005 * value
        self.reCalc()

    #delta_accl means with time going, dx increases by the value of delta_accl
    def gen_data(self,x0,dx,count,noise_factor,delta_accl=0.0):
        zs = []
        for i in range(count):
            zs.append(x0 + dx*i + randn()*noise_factor)
            dx += delta_accl
        return zs
        

    def g_h_filter(self,data, x0, dx, g, h, dt=1.):
        #The first estimation should be the same with state x 
        x_est = x0
        results = []
        #With each measurement data coming, do the prediction and update steps
        for z in data:
            # prediction step: need to prediction values for state and gain
            x_pred = x_est + (dx*dt)
            #The gain weight is varying actually, need some code here!!!
            dx = dx

            # update step
            residual = z - x_pred
            #update the gain according to measurement
            dx = dx + h * (residual) / dt
            #choose best estimation somewhere in between pre-estimation and measurement!
            x_est = x_pred + g * residual
            results.append(x_est)
        return np.array(results)

    def reCalc(self):
        print("reCalc triggered!")
        #del self.data[:]
        self.data = self.g_h_filter(data = self.measuredData,x0=self.filter_x,dx=self.filter_dx,g=self.filter_g,h=self.filter_h)
        self.plot()

''' End Class '''

if __name__== '__main__':
    app = QApplication(sys.argv)
    #QApplication.setStyle(QStyleFactory.create('Plastique'))
    myGUI = CustomMainWindow()


    sys.exit(app.exec_())

