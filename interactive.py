import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
from PyQt5 import QtGui
from PyQt5 import QtCore
#from PyQt5.QtWidgets import QMainWindow,QApplication
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget,QFrame,QMainWindow,QSizePolicy)
import time
import threading

class WidgetGallery(QMainWindow):
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)
        self.setGeometry(300, 300, 800, 400)
        self.setWindowTitle("h_g_filter window")
        self.x = 0
        self.dx = 0
        self.h = 0
        self.g = 0
        self.isRecal = False
        self.measuredData = self.gen_data(5,5,100,50)
        self.data = []
        self.FRAME_A = QFrame(self)
        self.FRAME_A.setStyleSheet("QWidget { background-color: %s }" % QtGui.QColor(210,210,235,255).name())
        self.mainLayout = QGridLayout()
        self.FRAME_A.setLayout(self.mainLayout)
        self.setCentralWidget(self.FRAME_A)


        self.createFilterGroupBox()
        #create Matplot figure 
        self.figure_canvas = Figure_Canvas()
        
        self.mainLayout.addWidget(self.filterGroupBox, *(0,0))
        self.mainLayout.addWidget(self.figure_canvas, *(0,1))
        self.setLayout(self.mainLayout)
        self.setWindowTitle("h_g_filter")
        # Add the callbackfunc to ..
        myDataLoop = threading.Thread(name = 'myDataLoop', target = self.dataSendLoop, daemon = True, args = (self.addData_callbackFunc,))
        myDataLoop.start()
        self.show()

    def addData_callbackFunc(self, value):
        # print("Add data: " + str(value))
        self.figure_canvas.addData(value)

    

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

        layout.setRowStretch(5, 1)
        self.filterGroupBox.setLayout(layout)
        self.show()

    def valueChanged_x(self):
        self.x = self.slider_x.value()*4 - 200
        self.xValue.setText(str(self.x))
        self.reCalc()

    def valueChange_dx(self):
        self.dx = self.slider_dx.value() - 50
        self.dxValue.setText(str(self.dx))
        self.reCalc()

    def valueChange_h(self):
        self.h = 0.0199 * self.slider_h.value() + 0.01
        self.hValue.setText(str(self.h))
        self.reCalc()

    def valueChange_g(self):
        self.g = 0.005 * self.slider_g.value()
        self.gValue.setText(str(self.g))
        self.reCalc()

    def createProgressBar(self):
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 10000)
        self.progressBar.setValue(0)

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
        del self.data[:]
        self.isRecal = True
        self.data = self.g_h_filter(data = self.measuredData,x0=self.x,dx=self.dx,g=self.g,h=self.h)
        self.isRecal = False

    def dataSendLoop(addData_callbackFunc):
        # Setup the signal-slot mechanism.
        mySrc = Communicate()
        mySrc.data_signal.connect(addData_callbackFunc)
        while(self.isRecal==False):
            if(i > len(self.data)):
                i = 0
            time.sleep(0.01)
            mySrc.data_signal.emit(self.data[i]) # <- Here you emit a signal!
            i += 1
        ###
    ###

''' End Class '''


# You need to setup a signal slot mechanism, to 
# send data to your GUI in a thread-safe way.
# Believe me, if you don't do this right, things
# go very very wrong..
class Communicate(QtCore.QObject):
    data_signal = QtCore.pyqtSignal(float)

''' End Class '''


class Figure_Canvas(FigureCanvas, TimedAnimation):#inheriting from the FigureCanvas can build the bridge between Qwidget and matplotlib of FigureCanvas   
    def __init__(self,xlim=100,ylim=1000):

        self.addedData = []
        print(matplotlib.__version__)
        # The data
        self.xlim = xlim
        self.ylim = ylim
        self.n = np.linspace(0, self.xlim - 1, self.xlim)

        # The window
        self.fig = Figure(figsize=(25,20), dpi=100)
        self.ax1 = self.fig.add_subplot(111)


        # self.ax1 settings
        self.ax1.set_xlabel('time')
        self.ax1.set_ylabel('y value')
        self.line1 = Line2D([], [], color='blue')
        self.ax1.add_line(self.line1)
        self.ax1.set_xlim(0, self.xlim - 1)
        self.ax1.set_ylim(0, self.ylim)


        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval = 50, blit = True)
    def new_frame_seq(self):
        return iter(range(self.n.size))

    def addData(self, value):
        print("Figure_Canvas:addData triggered!")
        self.addedData.append(value)
    def _init_draw(self):
        lines = [self.line1]
        for l in lines:
            l.set_data([], [])
    
    def _step(self, *args):
        # Extends the _step() method for the TimedAnimation class.
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            print(str(self.abc))
            TimedAnimation._stop(self)
            pass

    def _draw_frame(self, framedata):
        margin = 1
        while(len(self.addedData) > 0):
            self.y = np.roll(self.y, -1)
            self.y[-1] = self.addedData[0]
            del(self.addedData[0])


        self.line1.set_data(self.n, self.y)
        

    


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    gallery = WidgetGallery()
    gallery.showMaximized()
    sys.exit(app.exec_()) 
