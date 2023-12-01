import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
from UI import *
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtGui import QPixmap, QImage, QIcon

class MiApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowTitle("Clasificador de perros y gatos")
        self.setWindowIcon(QIcon('logo_app.ico'))

        self.ui.new_classification.hide()
        self.ui.upload.clicked.connect(self.load_image)
        self.ui.inference.clicked.connect(self.inference)
        self.ui.new_classification.clicked.connect(self.show_buttons)

    def show_buttons(self):
        self.ui.new_classification.hide()
        self.ui.upload.show()
        self.ui.inference.show()

    def show_image(self):

        self.cv_image = cv2.resize(self.cv_image, (180, 180))
        # Convertir la imagen de BGR a RGB (necesario para mostrar en PyQt)
        rgb_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)

        # Obtener las dimensiones de la imagen
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width

        # Crear un objeto QImage
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Crear un objeto QPixmap a partir de QImage
        pixmap = QPixmap.fromImage(q_image)

        # Mostrar la imagen en QLabel
        self.ui.image.setPixmap(pixmap)

    def load_image(self):
        # Abrir un cuadro de diálogo para seleccionar una imagen
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Imágenes (*.png *.jpg *.bmp)")
        file_dialog.setViewMode(QFileDialog.Detail)
        
        if file_dialog.exec_():
            # Obtener la ruta del archivo seleccionado
            file_path = file_dialog.selectedFiles()[0]

            # Cargar la imagen con OpenCV
            self.cv_image = cv2.imread(file_path)

            # Mostrar la nueva imagen en QLabel
            self.show_image()

    def inference(self):
        self.model_file = 'model.h5'
        self.model = keras.models.load_model(self.model_file)

        self.result = self.model.predict(np.expand_dims(self.cv_image, axis=0))

        self.score = float(self.result[0])
        self.ui.result.setText(f"Esta imagen es {100 * (1 - self.score):.2f}% gato y {100 * self.score:.2f}% perro.")
        self.ui.upload.hide()
        self.ui.inference.hide()
        self.ui.new_classification.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mi_app = MiApp()
    mi_app.show()
    sys.exit(app.exec_())