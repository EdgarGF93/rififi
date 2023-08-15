

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLineEdit, QLabel, QSpinBox
from silx.gui.plot.PlotWindow import Plot1D, Plot2D
import numpy as np
import sys


LABEL_DT = "Temp. resolution (s)"
LABEL_FREQ_1 = "Freq. 1 (Hz)"
LABEL_FREQ_2 = "Freq. 2 (Hz)"
LABEL_FREQ_3 = "Freq. 3 (Hz)"
LABEL_FREQ_4 = "Freq. 4 (Hz)"

LABEL_T0 = "T0 (s)"
LABEL_TF = "Tf (s)"
LABEL_NOISE_1 = "Noise amplitude 1"
LABEL_NOISE_2 = "Noise amplitude 2"

DT_DEFAULT = "0.001"
FREQ1_DEFAULT = int(5)
FREQ2_DEFAULT = int(0)
FREQ3_DEFAULT = int(0)
FREQ4_DEFAULT = int(0)
NOISE1_DEFAULT = int(0)
NOISE2_DEFAULT = int(0)
T0_DEFAULT = int(0)
TF_DEFAULT = int(1)

class RFFWidgetLayOut(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._build()

    def _build(self):
        # Global Grid
        self.gridlayout = QGridLayout()
        self.setLayout(self.gridlayout)

        # Main 3x3 Grids
        self.main_grid_11 = QGridLayout()
        self.main_grid_12 = QGridLayout()
        self.main_grid_13 = QGridLayout()
        self.main_grid_21 = QGridLayout()
        self.main_grid_22 = QGridLayout()
        self.main_grid_23 = QGridLayout()
        self.main_grid_31 = QGridLayout()
        self.main_grid_32 = QGridLayout()
        self.main_grid_33 = QGridLayout()

        self.gridlayout.setRowStretch(1,1)
        self.gridlayout.setRowStretch(2,3)
        self.gridlayout.setRowStretch(3,1)
        self.gridlayout.setColumnStretch(1,1)
        self.gridlayout.setColumnStretch(2,3)
        self.gridlayout.setColumnStretch(3,1)

        self.gridlayout.addLayout(self.main_grid_11, 1, 1)
        self.gridlayout.addLayout(self.main_grid_12, 1, 2)
        self.gridlayout.addLayout(self.main_grid_13, 1, 3)
        self.gridlayout.addLayout(self.main_grid_21, 2, 1)
        self.gridlayout.addLayout(self.main_grid_22, 2, 2)
        self.gridlayout.addLayout(self.main_grid_23, 2, 3)
        self.gridlayout.addLayout(self.main_grid_31, 3, 1)
        self.gridlayout.addLayout(self.main_grid_32, 3, 2)
        self.gridlayout.addLayout(self.main_grid_33, 3, 3)

        # Input Frequencies and Times
        grid = self.main_grid_21
        self.label_dt = QLabel(LABEL_DT)
        self.lineedit_dt = QLineEdit(DT_DEFAULT)
        self.label_t0 = QLabel(LABEL_T0)
        self.spinbox_t0 = QSpinBox()
        self.spinbox_t0.setValue(T0_DEFAULT)
        self.label_tf = QLabel(LABEL_TF)
        self.spinbox_tf = QSpinBox()
        self.spinbox_tf.setValue(TF_DEFAULT)
        self.label_freq_1 = QLabel(LABEL_FREQ_1)
        self.label_freq_2 = QLabel(LABEL_FREQ_2)
        self.label_freq_3 = QLabel(LABEL_FREQ_3)
        self.label_freq_4 = QLabel(LABEL_FREQ_4)
        self.spinbox_freq_1 = QSpinBox()
        self.spinbox_freq_1.setValue(FREQ1_DEFAULT)
        self.spinbox_freq_2 = QSpinBox()
        self.spinbox_freq_2.setValue(FREQ2_DEFAULT)
        self.spinbox_freq_3 = QSpinBox()
        self.spinbox_freq_3.setValue(FREQ3_DEFAULT)
        self.spinbox_freq_4 = QSpinBox()
        self.spinbox_freq_4.setValue(FREQ4_DEFAULT)
        self.label_noise_1 = QLabel(LABEL_NOISE_1)
        self.label_noise_2 = QLabel(LABEL_NOISE_2)
        self.spinbox_noise_1 = QSpinBox()
        self.spinbox_noise_1.setValue(NOISE1_DEFAULT)
        self.spinbox_noise_2 = QSpinBox()
        self.spinbox_noise_2.setValue(NOISE2_DEFAULT)

        grid.addWidget(self.label_dt, 1, 1)
        grid.addWidget(self.lineedit_dt, 1, 2)
        grid.addWidget(self.label_t0, 2, 1)
        grid.addWidget(self.spinbox_t0, 2, 2)
        grid.addWidget(self.label_tf, 3, 1)
        grid.addWidget(self.spinbox_tf, 3, 2)
        grid.addWidget(self.label_freq_1, 4, 1)
        grid.addWidget(self.spinbox_freq_1, 4, 2)
        grid.addWidget(self.label_freq_2, 5, 1)
        grid.addWidget(self.spinbox_freq_2, 5, 2)
        grid.addWidget(self.label_freq_3, 6, 1)
        grid.addWidget(self.spinbox_freq_3, 6, 2)
        grid.addWidget(self.label_freq_4, 7, 1)
        grid.addWidget(self.spinbox_freq_4, 7, 2)
        grid.addWidget(self.label_noise_1, 8, 1)
        grid.addWidget(self.spinbox_noise_1, 8, 2)
        grid.addWidget(self.label_noise_2, 9, 1)
        grid.addWidget(self.spinbox_noise_2, 9, 2)

        # Central Plot
        grid = self.main_grid_22
        self.graph = Plot1D()
        grid.addWidget(self.graph)

        # FFT plot
        grid = self.main_grid_23
        self.graph_fft = Plot1D()
        grid.addWidget(self.graph_fft)






class RFFWidget(RFFWidgetLayOut):
    def __init__(self) -> None:
        super().__init__()
        self.N = 0
        self.update_signal()
        self.update_fft()
        self._build_callbacks()

    def _build_callbacks(self):
        self.spinbox_freq_1.valueChanged.connect(
            lambda : (
            self.update_signal(),
            self.update_fft(),
            )
        )
        self.spinbox_freq_2.valueChanged.connect(
            lambda : (
            self.update_signal(),
            self.update_fft(),
            )
        )
        self.spinbox_freq_3.valueChanged.connect(
            lambda : (
            self.update_signal(),
            self.update_fft(),
            )
        )
        self.spinbox_freq_4.valueChanged.connect(
            lambda : (
            self.update_signal(),
            self.update_fft(),
            )
        )
        self.spinbox_noise_1.valueChanged.connect(
            lambda : (
            self.update_signal(),
            self.update_fft(),
            )
        )
        self.spinbox_noise_2.valueChanged.connect(
            lambda : (
            self.update_signal(),
            self.update_fft(),
            )
        )
        self.spinbox_t0.valueChanged.connect(
            lambda : (
            self.update_signal(),
            self.update_fft(),
            )
        )
        self.spinbox_tf.valueChanged.connect(
            lambda : (
            self.update_signal(),
            self.update_fft(),
            )
        )

    def update_signal(self):
        dt = self.lineedit_dt.text()
        dt = float(dt)
        t0 = float(self.spinbox_t0.value())
        tf = float(self.spinbox_tf.value())
        f1 = float(self.spinbox_freq_1.value())
        f2 = float(self.spinbox_freq_2.value())
        f3 = float(self.spinbox_freq_3.value())
        f4 = float(self.spinbox_freq_4.value())
        n1 = float(self.spinbox_noise_1.value())
        n2 = float(self.spinbox_noise_2.value())

        time_vector = np.arange(t0, tf, dt)
        self.N = len(time_vector)

        freqs = np.array([f1, f2, f3, f4])
        noises = np.array([n1, n2])
        
        noise_functions = np.array([np.random.random(size=self.N) * amp * 2 - amp for amp in noises]) 
        noise_total = np.sum((noise_functions), axis=0)

        y_functions = np.array([np.sin(2 * np.pi * f * time_vector) for f in freqs])
        y_clean = np.sum((y_functions), axis=0)
        y_noisy = y_clean + noise_total
        self.signal = y_noisy

        self.graph.addCurve(
            x=time_vector,
            y=y_noisy,
            legend=f"{11}",
            resetzoom=True,
        )
        self.graph.setLimits(
            xmin=t0,
            xmax=tf,
            ymin=self.graph.getGraphYLimits()[0],
            ymax=self.graph.getGraphYLimits()[1],
        )

    def update_fft(self):
        fhat = np.fft.fft(self.signal, self.N)
        PSD = (fhat * np.conj(fhat)) / self.N
        dt = self.lineedit_dt.text()
        dt = float(dt)
        freq_vector = np.fft.fftfreq(n=self.N, d=dt)

        self.graph_fft.addCurve(
            x=freq_vector,
            y=PSD,
            legend=f"{11}",
            resetzoom=True,
        )


class RFFWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._guiwidget = RFFWidget()
        self.setCentralWidget(self._guiwidget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RFFWindow()

    window.show()
    app.exec_()