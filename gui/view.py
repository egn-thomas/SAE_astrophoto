from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
    QGroupBox, QGridLayout, QSplitter
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt
import numpy as np

class ImageComparisonWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.processed_pixmap = None
        self.split_position = 0.5  # 0.0 to 1.0
        self.setMinimumHeight(400)
        self.setStyleSheet("background-color: #111; border: 1px solid #333;")

    def set_images(self, original_image, processed_image):
        self.original_pixmap = self.convert_to_pixmap(original_image)
        self.processed_pixmap = self.convert_to_pixmap(processed_image)
        self.update()

    def convert_to_pixmap(self, image):
        if image is None:
            return None
        image = np.ascontiguousarray(image)
        if image.ndim == 3:
            h, w, c = image.shape
            q_img = QImage(image.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            h, w = image.shape
            q_img = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(q_img)

    def set_split_position(self, position):
        self.split_position = position / 100.0  # assuming slider 0-100
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.processed_pixmap and self.original_pixmap:
            rect = self.rect()
            
            # Scale both pixmaps to fit the widget while maintaining aspect ratio
            scaled_processed = self.processed_pixmap.scaled(rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scaled_original = self.original_pixmap.scaled(rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Get the actual size of the scaled images
            processed_size = scaled_processed.size()
            original_size = scaled_original.size()
            
            # Calculate positions to center the images
            processed_x = (rect.width() - processed_size.width()) // 2
            processed_y = (rect.height() - processed_size.height()) // 2
            original_x = (rect.width() - original_size.width()) // 2
            original_y = (rect.height() - original_size.height()) // 2
            
            # Calculate split position
            split_x = int(self.split_position * rect.width())
            
            # Draw processed image on the left
            painter.drawPixmap(processed_x, processed_y, scaled_processed, 0, 0, split_x - processed_x, processed_size.height())
            
            # Draw original image on the right
            painter.drawPixmap(split_x, original_y, scaled_original, split_x - original_x, 0, rect.width() - split_x, original_size.height())
            
            # Draw split line
            painter.setPen(QPen(Qt.white, 2))
            painter.drawLine(split_x, 0, split_x, rect.height())

class View(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Selective Erosion GUI')
        self.setGeometry(200, 150, 1000, 700)

        #  STYLE GLOBAL 
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #dddddd;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3a3f44;
                padding: 10px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #50565c;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666666;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #444;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00aaff;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QGroupBox {
                border: 1px solid #444;
                margin-top: 12px;
                padding: 10px;
                border-radius: 6px;
                font-weight: bold;
            }
        """)

        main_layout = QVBoxLayout()

        #  BOUTONS EN HAUT 
        buttons_layout = QHBoxLayout()
        
        self.load_button = QPushButton('Charger un fichier FITS')
        self.load_button.setFixedHeight(40)
        buttons_layout.addWidget(self.load_button)

        self.mode_button = QPushButton("Mode: Multitaille")
        self.mode_button.setFixedHeight(40)
        self.mode_button.setStyleSheet("""
            QPushButton {
                background-color: #2a5a2a;
                padding: 10px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #3a7a3a;
            }
        """)
        buttons_layout.addWidget(self.mode_button)
        self.ai_button = QPushButton("IA: Désactivée")
        self.ai_button.setStyleSheet("background-color: #444; color: white; padding: 10px; border-radius: 6px;")
        buttons_layout.addWidget(self.ai_button)

        self.save_button = QPushButton('Télécharger l\'image')
        self.save_button.setFixedHeight(40)
        self.save_button.setEnabled(False)  # Désactivé par défaut
        buttons_layout.addWidget(self.save_button)
        
        main_layout.addLayout(buttons_layout)

        #  GROUPE DES SLIDERS 
        sliders_group = QGroupBox("Paramètres du traitement")
        sliders_layout = QGridLayout()

        # Kernel size
        self.kernel_label = QLabel('Taille du noyau : 21')
        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setMinimum(5)
        self.kernel_slider.setMaximum(51)
        self.kernel_slider.setValue(21)
        self.kernel_slider.setTickInterval(2)
        sliders_layout.addWidget(self.kernel_label, 0, 0)
        sliders_layout.addWidget(self.kernel_slider, 0, 1)

        # Threshold
        self.threshold_label = QLabel('Seuil du masque : 0.5')
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(20)
        self.threshold_slider.setValue(5)
        sliders_layout.addWidget(self.threshold_label, 1, 0)
        sliders_layout.addWidget(self.threshold_slider, 1, 1)

        # Blur sigma
        self.blur_sigma_label = QLabel('Sigma du flou : 5')
        self.blur_sigma_slider = QSlider(Qt.Horizontal)
        self.blur_sigma_slider.setMinimum(1)
        self.blur_sigma_slider.setMaximum(20)
        self.blur_sigma_slider.setValue(5)
        sliders_layout.addWidget(self.blur_sigma_label, 2, 0)
        sliders_layout.addWidget(self.blur_sigma_slider, 2, 1)

        # Mask dilate
        self.mask_dilate_label = QLabel('Taille dilatation : 5')
        self.mask_dilate_slider = QSlider(Qt.Horizontal)
        self.mask_dilate_slider.setMinimum(1)
        self.mask_dilate_slider.setMaximum(15)
        self.mask_dilate_slider.setValue(5)
        sliders_layout.addWidget(self.mask_dilate_label, 3, 0)
        sliders_layout.addWidget(self.mask_dilate_slider, 3, 1)

        # Attenuation
        self.attenuation_label = QLabel('Atténuation : 0.4')
        self.attenuation_slider = QSlider(Qt.Horizontal)
        self.attenuation_slider.setMinimum(0)
        self.attenuation_slider.setMaximum(10)
        self.attenuation_slider.setValue(4)
        sliders_layout.addWidget(self.attenuation_label, 4, 0)
        sliders_layout.addWidget(self.attenuation_slider, 4, 1)

        sliders_group.setLayout(sliders_layout)
        main_layout.addWidget(sliders_group)

        #  CURSEUR POUR COMPARAISON AVANT/APRÈS 
        self.comparison_slider = QSlider(Qt.Horizontal)
        self.comparison_slider.setMinimum(0)
        self.comparison_slider.setMaximum(100)
        self.comparison_slider.setValue(50)
        self.comparison_slider.setFixedHeight(30)
        main_layout.addWidget(self.comparison_slider)

        #  AFFICHAGE DE L'IMAGE 
        self.image_widget = ImageComparisonWidget()

        # Utiliser un splitter pour ajuster les proportions en plein écran
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(sliders_group)
        self.splitter.addWidget(self.image_widget)

        main_layout.addWidget(self.splitter)

        self.setLayout(main_layout)

    #  MISE À JOUR DE L'IMAGE 
    def update_image(self, image):
        # This method is now for setting the processed image
        # The widget will handle displaying both
        pass

    def set_comparison_images(self, original_image, processed_image):
        self.image_widget.set_images(original_image, processed_image)

    #  MISE À JOUR DES LABELS DES SLIDERS 
    def update_labels(self, kernel_size, threshold, blur_sigma, mask_dilate_size, attenuation_factor):
        self.kernel_label.setText(f'Taille du noyau : {kernel_size}')
        self.threshold_label.setText(f'Seuil du masque : {threshold:.1f}')
        self.blur_sigma_label.setText(f'Sigma du flou : {blur_sigma}')
        self.mask_dilate_label.setText(f'Taille dilatation : {mask_dilate_size}')
        self.attenuation_label.setText(f'Atténuation : {attenuation_factor:.1f}')

    def resizeEvent(self, event):
        if self.isFullScreen():
            total_height = self.height()
            image_height = int(total_height * 0.75)
            sliders_height = total_height - image_height
            self.splitter.setSizes([sliders_height, image_height])
        super().resizeEvent(event)